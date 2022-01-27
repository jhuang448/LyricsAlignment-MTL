import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import time
from tqdm import tqdm

from data import get_dali_folds, LyricsAlignDataset
from test import validate
from utils import worker_init_fn, load_model

from model import AcousticModel, data_processing, MultiTaskLossWrapper

def train(model, device, train_loader, criterion, optimizer, batch_size, model_type, loss_w=0.1):
    avg_time = 0.
    model.train()
    data_len = len(train_loader.dataset) // batch_size

    if model_type == "MTL":
        train_loss = 0.
        train_loss_phone = 0.
        train_loss_melody = 0.

        with tqdm(total=data_len) as pbar:
            for batch_idx, _data in enumerate(train_loader):
                spectrograms, phones, input_lengths, phone_lengths, pcs = _data
                spectrograms, phones, pcs = spectrograms.to(device), phones.to(device), pcs.to(device)

                t = time.time()

                optimizer.zero_grad()

                output = model(spectrograms)  # (batch, time, n_class)
                phone_gt = (phones, input_lengths, phone_lengths)
                melody_gt = pcs

                loss_phone, loss_melody = criterion(output, phone_gt, melody_gt)
                loss = loss_phone + loss_melody * loss_w
                loss.backward()

                optimizer.step()

                t = time.time() - t
                avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

                train_loss += loss.item()
                train_loss_phone += loss_phone.item()
                train_loss_melody += loss_melody.item()

                pbar.set_description("Current loss: {:.4f}".format(loss))
                pbar.update(1)

                if batch_idx == data_len:
                    break

        return train_loss / data_len, train_loss_phone / data_len, train_loss_melody / data_len
    else: # baseline
        train_loss = 0.

        with tqdm(total=data_len) as pbar:
            for batch_idx, _data in enumerate(train_loader):
                spectrograms, phones, input_lengths, phone_lengths, pcs = _data
                spectrograms, phones, pcs = spectrograms.to(device), phones.to(device), pcs.to(device)

                t = time.time()

                optimizer.zero_grad()

                output_phone = model(spectrograms)  # (batch, time, n_class)
                output_phone = F.log_softmax(output_phone, dim=2)
                output_phone = output_phone.transpose(0, 1)  # (time, batch, n_class)

                loss = criterion(output_phone, phones, input_lengths, phone_lengths)
                loss.backward()

                optimizer.step()

                t = time.time() - t
                avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

                train_loss += loss.item()

                pbar.set_description("Current loss: {:.4f}".format(loss))
                pbar.update(1)

                if batch_idx == data_len:
                    break

        return train_loss / data_len, train_loss / data_len, None

def main(args):

    if args.model == "baseline":
        n_class = 41
    elif args.model == "MTL":
        n_class = (41, 47)
    else:
        ValueError("Invalid model type.")

    hparams = {
        "n_cnn_layers": args.cnn_layers,
        "n_rnn_layers": 3,
        "rnn_dim": args.rnn_dim,
        "n_class": n_class, # (phone, pitch) or phone
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1,
        "learning_rate": args.lr,
        "input_sample": args.input_sample,
        "batch_size": args.batch_size
    }

    # set CUDA
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda and args.cuda else "cpu")
    print(device)

    # create folders for checkpoints and logs
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # init model
    model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'], \
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    # prepare dataset
    if os.path.exists(os.path.join(args.hdf_dir, "val.hdf5")) and os.path.exists(os.path.join(args.hdf_dir, "train.hdf5")):
        dali_split = {"train": [], "val": []}  # h5 files already saved
    else:
        # call the DALI wrapper to get word-level annotations
        dali_split = get_dali_folds(args.dataset_dir, args.sepa_dir)

    val_data = LyricsAlignDataset(dali_split, "val", args.sr, hparams['input_sample'], args.hdf_dir, dummy=args.dummy)
    train_data = LyricsAlignDataset(dali_split, "train", args.sr, hparams['input_sample'], args.hdf_dir, dummy=args.dummy)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=hparams["batch_size"],
                                   shuffle=True,
                                   worker_init_fn=worker_init_fn,
                                   collate_fn=lambda x: data_processing(x),
                                   **kwargs)
    val_loader = data.DataLoader(dataset=val_data,
                                   batch_size=hparams["batch_size"],
                                   shuffle=False,
                                   collate_fn=lambda x: data_processing(x),
                                   **kwargs)

    optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])
    if args.model == "baseline":
        criterion = nn.CTCLoss(blank=40, zero_infinity=True)
    else:
        criterion = MultiTaskLossWrapper()

    # training state dict for saving checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf,
             "best_checkpoint": None}

    # load a pre-trained model
    if args.load_model is not None:
        state = load_model(model, args.load_model, args.cuda)

    from torch.utils.tensorboard import SummaryWriter
    import datetime
    current = datetime.datetime.now()
    writer = SummaryWriter(os.path.join(args.log_dir + current.strftime("%m:%d:%H:%M")))

    while state["worse_epochs"] < 20:
        print("Training one epoch from epoch " + str(state["epochs"]))

        # train
        train_loss, train_loss_phone, train_loss_melody = train(model, device, train_loader, criterion, optimizer, args.batch_size, args.model, args.loss_w)
        print("TRAINING FINISHED: LOSS: " + str(train_loss) + " phone loss: " + str(train_loss_phone) + " melody loss: " + str(train_loss_melody))
        writer.add_scalar("train/epoch_loss", train_loss, state["epochs"])
        writer.add_scalar("train/phone_loss", train_loss_phone, state["epochs"])
        if args.model == "MTL":
            writer.add_scalar("train/melody_loss", train_loss_melody, state["epochs"])

        val_loss, val_loss_phone, val_loss_melody = validate(args.batch_size, model, -1, criterion, val_loader, device, args.model, args.loss_w)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss) + " phone loss: " + str(val_loss_phone) + " melody loss: " + str(val_loss_melody))
        writer.add_scalar("val/loss", val_loss, state["epochs"])
        writer.add_scalar("val/phone_loss", val_loss_phone, state["epochs"])
        if args.model == "MTL":
            writer.add_scalar("val/melody_loss", val_loss_melody, state["epochs"])

        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["epochs"]))
        if val_loss >= state["best_loss"]:
            if state["epochs"] >= 20:  # after 20 epochs, start early stopping counts
                state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        # save all checkpoints
        print("Saving model... best_epoch {} best_loss {}".format(state["best_checkpoint"], state["best_loss"]))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'state': state
        }, checkpoint_path)

        state["epochs"] += 1

    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy train/val sets (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset path')
    parser.add_argument('--sepa_dir', type=str, required=True,
                        help='Where all the separated vocals are stored.')
    parser.add_argument('--hdf_dir', type=str, default="./hdf/",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Folder to write checkpoints into')
    parser.add_argument('--model', type=str, default="baseline",
                        help='"baseline" or "MTL"')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=22050,
                        help="Sampling rate")
    parser.add_argument('--input_sample', type=int, default=123904,
                        help="Input sample")
    parser.add_argument('--cnn_layers', type=int, default=1,
                        help="num of cnn layers")
    parser.add_argument('--rnn_dim', type=int, default=256,
                        help="dimension of rnn layers")
    parser.add_argument('--loss_w', type=float, default=0.1,
                        help="weight of voc loss")

    args = parser.parse_args()
    print(args)

    main(args)