import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import warnings

from utils import notes_to_pc

# following FFT parameters are designed for a 22.5k sampling rate
sr = 22050
n_fft = 512
resolution = 256/22050*3

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128, n_fft=n_fft),
    )

def data_processing(data):
    spectrograms = []
    phones = []
    pcs = []
    input_lengths = []
    phone_lengths = []
    for (waveform, _, _, phone, notes) in data:
        waveform = torch.Tensor(waveform)
        # convert to Mel
        spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1) # time x n_mels
        spectrograms.append(spec)

        # get phoneme list (mapped to integers)
        phone = torch.Tensor(phone)
        phones.append(phone)

        # get the pitch contour
        # the number 3 here and below is due the the maxpooling along the frequency axis
        pc = notes_to_pc(notes, resolution, spec.shape[0] // 3)
        pcs.append(pc)

        input_lengths.append(spec.shape[0]//3)
        phone_lengths.append(len(phone))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    phones = nn.utils.rnn.pad_sequence(phones, batch_first=True)

    return spectrograms, phones, input_lengths, phone_lengths, torch.LongTensor(pcs)

class CNNLayerNorm(nn.Module):
    '''Layer normalization built for cnns input'''

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    '''Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    '''

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalLSTM(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.BiLSTM(x)
        x = self.dropout(x)
        return x

class AcousticModel(nn.Module):
    '''
        The acoustic model: baseline and MTL share the same class,
        the only difference is the target dimension of the last fc layer
    '''

    def __init__(self, n_cnn_layers, rnn_dim, n_class, n_feats, stride=1, dropout=0.1):
        super(AcousticModel, self).__init__()

        self.n_class = n_class
        if isinstance(n_class, int):
            target_dim = n_class
        else:
            target_dim = n_class[0] * n_class[1]

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, stride=stride, padding=3 // 2),
            nn.ReLU()
        )

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(n_feats, n_feats, kernel=3, stride=1, dropout=dropout, n_feats=128)
            for _ in range(n_cnn_layers)
        ])

        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 3))
        self.fully_connected = nn.Linear(n_feats * 64, rnn_dim)

        self.bilstm = nn.Sequential(
            BidirectionalLSTM(rnn_dim=rnn_dim, hidden_size=rnn_dim, dropout=dropout, batch_first=True),
            BidirectionalLSTM(rnn_dim=rnn_dim * 2, hidden_size=rnn_dim, dropout=dropout, batch_first=False),
            BidirectionalLSTM(rnn_dim=rnn_dim * 2, hidden_size=rnn_dim, dropout=dropout, batch_first=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, target_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.rescnn_layers(x)
        x = self.maxpooling(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)

        x = self.bilstm(x)
        x = self.classifier(x)

        if isinstance(self.n_class, tuple):
            x = x.view(sizes[0], sizes[3], self.n_class[0], self.n_class[1])

        return x

class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()

        self.criterion_lyrics = nn.CTCLoss(blank=40, zero_infinity=True)
        self.criterion_melody = nn.CrossEntropyLoss()

    def forward(self, mat3d, lyrics_gt, melody_gt):

        n_batch, n_frame, n_ch, n_p = mat3d.shape # (batch, time, phone, pitch)

        y_lyrics = torch.sum(mat3d, dim=3) # (batch, time, n_ch)
        y_melody = torch.sum(mat3d, dim=2) # (batch, time, n_p)

        y_lyrics = F.log_softmax(y_lyrics, dim=2)
        y_lyrics = y_lyrics.transpose(0, 1) # (time, batch, n_ch) reshape for CTC
        labels, input_lengths, label_lengths = lyrics_gt
        loss_lyrics = self.criterion_lyrics(y_lyrics, labels, input_lengths, label_lengths)

        y_melody = y_melody.transpose(1, 2)  # (batch, n_p, time)
        loss_melody = self.criterion_melody(y_melody, melody_gt)

        return loss_lyrics, loss_melody


class BoundaryDetection(nn.Module):

    def __init__(self, n_cnn_layers, rnn_dim, n_class, n_feats, stride=1, dropout=0.1):
        super(BoundaryDetection, self).__init__()

        self.n_class = n_class

        # n residual cnn layers with filter size of 32
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, stride=stride, padding=3 // 2),
            nn.ReLU()
        )

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(n_feats, n_feats, kernel=3, stride=1, dropout=dropout, n_feats=128)
            for _ in range(n_cnn_layers)
        ])

        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 3))
        self.fully_connected = nn.Linear(n_feats * 64, rnn_dim) # add a linear layer

        self.bilstm_layers = nn.Sequential(
            BidirectionalLSTM(rnn_dim=rnn_dim, hidden_size=rnn_dim, dropout=dropout, batch_first=True),
            BidirectionalLSTM(rnn_dim=rnn_dim * 2, hidden_size=rnn_dim, dropout=dropout, batch_first=False),
            BidirectionalLSTM(rnn_dim=rnn_dim * 2, hidden_size=rnn_dim, dropout=dropout, batch_first=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, n_class)  # birnn returns rnn_dim*2
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.rescnn_layers(x)
        x = self.maxpooling(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)

        x = self.bilstm_layers(x)

        x = self.classifier(x)
        x = x.view(sizes[0], sizes[3], self.n_class)

        x = torch.sigmoid(x)

        return x