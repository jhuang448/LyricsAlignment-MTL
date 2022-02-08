import h5py
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import string
from tqdm import tqdm
import logging

import DALI as dali_code
from utils import load, load_lyrics, gen_phone_gt, ToolFreq2Midi

phone_dict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
             'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
             'Z', 'ZH', ' ']
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}

def getDALI(database_path, vocal_path, lang, genre):
    dali_annot_path = os.path.join(database_path, 'annot_tismir')
    dali_audio_path = os.path.join(database_path, 'audio')
    dali_data = dali_code.get_the_DALI_dataset(dali_annot_path, skip=[], keep=[])

    # get audio list
    audio_list = os.listdir(os.path.join(dali_audio_path))

    subset = list()
    total_line_num = 0
    discard_line_num = 0

    for file in audio_list:
        if file.endswith('.mp3') and os.path.exists(os.path.join(dali_annot_path, file[:-4] + '.gz')):
            # get annotation for the current song
            try:
                entry = dali_data[file[:-4]].annotations['annot']
                entry_info = dali_data[file[:-4]].info

                # language filter
                if lang is not None and entry_info['metadata']['language'] != lang:
                    continue
                # genre filter
                if genre is not None and genre not in entry_info['metadata']['genres']:
                    continue

                song = {"id": file[:-4], "words": [], \
                        "path": os.path.join(dali_audio_path, file), \
                        "vocal_path": os.path.join(vocal_path, file[:-4] + "_vocals.mp3")}

                # notes
                notes_raw = entry["notes"]
                notes = [{"pitch": ToolFreq2Midi(note_raw['freq'][0]), "time": note_raw['time']} for note_raw in
                         notes_raw]
                song["notes"] = notes

                # words
                samples = entry["words"]
                words = []
                for sample in samples:
                    sample["duration"] = sample["time"][1] - sample["time"][0]

                    if sample["duration"] > 10.22: # remove words which are too long
                        # print(sample)
                        discard_line_num += 1

                    words.append(sample)

                    total_line_num += 1
                song["words"] = words

                # phoneme
                max_phone = -1
                phonemes_raw = entry["phonemes"]
                # phonemes_encode = [[s.encode() for s in sample["text"]] for sample in phonemes_raw]
                phonemes_encode = []
                for sample in phonemes_raw:
                    if len(sample["text"]) > max_phone:
                        max_phone = len(sample["text"])
                    phonemes_sample = []
                    for s in sample["text"]:
                        # if s not in phones_dict:
                        #     phones_dict.append(s)
                        phonemes_sample.append(s.encode())
                    phonemes_encode.append(phonemes_sample)
                song["phonemes"] = phonemes_encode
                song["max_phone"] = max_phone
                song["phone_num"] = len(phonemes_raw)

                # boundary
                song["paragraphs"] = entry["paragraphs"]
                song["lines"] = entry["lines"]

                subset.append(song)

                logging.debug("Successfully loaded {} songs".format(len(subset)))
            except:
                logging.warning("Error loading annotation for song {}".format(file))
                pass

    logging.debug("Scanning {} songs.".format(len(subset)))
    logging.debug("Total line num: {} Discarded line num: {}".format(total_line_num,  discard_line_num))

    return np.array(subset, dtype=object)

def get_dali_folds(database_path, vocal_path, lang="english", genre=None):
    dataset = getDALI(database_path, vocal_path, lang, genre)

    total_len = len(dataset)
    train_len = np.int(0.8 * total_len)

    train_list = np.random.choice(dataset, train_len, replace=False)
    val_list = [elem for elem in dataset if elem not in train_list]

    # dummy testing
    train_list = train_list[:20]
    val_list = val_list[:20]

    logging.debug("First training song: " + str(train_list[0]["id"]) + " " + str(len(train_list[0]["words"])) + " lines")
    logging.debug("train_list {} songs val_list {} songs".format(len(train_list), len(val_list)))
    return {"train" : train_list, "val" : val_list}

class LyricsAlignDataset(Dataset):
    def __init__(self, dataset, partition, sr, input_sample, hdf_dir, in_memory=False, dummy=False):
        '''

        :param dataset:     a list of song with line level annotation
        :param partition:   "train" or "val"
        :param sr:          sampling rate
        :param input_sample:      input and output length in samples
        :param hdf_dir:     hdf5 file
        :param in_memory:   load in memory or not
        :param dummy:       use a subset
        '''

        super(LyricsAlignDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        if dummy == False:
            self.hdf_file = os.path.join(hdf_dir, partition + ".hdf5")
        else:
            self.hdf_file = os.path.join(hdf_dir, partition + "_dummy.hdf5")

        self.sr = sr
        self.input_sample = input_sample
        self.hop = (input_sample // 2)
        self.in_memory = in_memory

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_file):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load song
                    y, _ = load(example["vocal_path"], sr=self.sr, mono=True)

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["audio_name"] = example["id"]
                    grp.attrs["input_length"] = y.shape[1]

                    # word level annotation
                    annot_num = len(example["words"])
                    lyrics = [sample["text"].encode() for sample in example["words"]]
                    times = np.array([sample["time"] for sample in example["words"]])

                    # note level annotation
                    notes = np.array(example["notes"])
                    note_num = len(notes)
                    pitches = np.array([int(note["pitch"]) for note in notes])
                    note_times = np.array([np.array([note['time'][0], note['time'][1]]) for note in notes])

                    # line
                    lines = example["lines"]
                    lines_times = np.array([np.array([l['time'][0], l['time'][1]]) for l in lines])

                    # phoneme
                    phonemes_encode = example["phonemes"]
                    max_phone = example["max_phone"]

                    grp.attrs["annot_num"] = annot_num
                    grp.attrs["note_num"] = note_num

                    # words and corresponding times
                    grp.create_dataset("lyrics", shape=(annot_num, 1), dtype='S100', data=lyrics)
                    grp.create_dataset("times", shape=(annot_num, 2), dtype=times.dtype, data=times)

                    # notes and corresponding times
                    grp.create_dataset("pitches", shape=(note_num, 1), dtype=np.short, data=pitches)
                    grp.create_dataset("note_times", shape=(note_num, 2), dtype=note_times.dtype, data=note_times)

                    grp.create_dataset("line_times", shape=(len(lines), 2), dtype=lines_times.dtype, data=lines_times)
                    grp.create_dataset("phonemes", shape=(annot_num, max_phone), dtype='S2')
                    for i in range(annot_num):
                        phonemes_sample = phonemes_encode[i]
                        grp["phonemes"][i, :len(phonemes_sample)] = np.array(phonemes_sample)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected.")

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:

            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [( (l - input_sample) // self.hop) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):

        # open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        while True:
            # Loop until it finds a valid sample

            # Find out which slice of targets we want to read
            song_idx = self.start_pos.bisect_right(index)
            if song_idx > 0:
                index = index - self.start_pos[song_idx - 1]

            # length of audio signal
            audio_length = self.hdf_dataset[str(song_idx)].attrs["input_length"]
            # number of words in this song
            annot_num = self.hdf_dataset[str(song_idx)].attrs["annot_num"]

            # determine where to start
            start_pos = index * self.hop
            end_pos = start_pos + self.input_sample

            # front padding
            if start_pos < 0:
                # Pad manually since audio signal was too short
                pad_front = abs(start_pos)
                start_pos = 0
            else:
                pad_front = 0

            # back padding
            if end_pos > audio_length:
                # Pad manually since audio signal was too short
                pad_back = end_pos - audio_length
                end_pos = audio_length
            else:
                pad_back = 0

            # read audio and zero padding
            audio = self.hdf_dataset[str(song_idx)]["inputs"][0, start_pos:end_pos].astype(np.float32)
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            # find the lyrics within (start_target_pos, end_target_pos)
            words_start_end_pos = self.hdf_dataset[str(song_idx)]["times"][:]
            try:
                first_word_to_include = next(x for x, val in enumerate(list(words_start_end_pos[:, 0]))
                                             if val > start_pos/self.sr)
            except StopIteration:
                first_word_to_include = np.Inf

            try:
                last_word_to_include = annot_num - 1 - next(x for x, val in enumerate(reversed(list(words_start_end_pos[:, 1])))
                                             if val < end_pos/self.sr)
            except StopIteration:
                last_word_to_include = -np.Inf

            # find the notes within (start_target_pos, end_target_pos)
            note_onsets = SortedList(self.hdf_dataset[str(song_idx)]["note_times"][:, 1])
            first_note_to_include = note_onsets.bisect_right(start_pos / self.sr)
            note_offsets = SortedList(self.hdf_dataset[str(song_idx)]["note_times"][:, 0])
            last_note_to_include = note_offsets.bisect_left(end_pos / self.sr) - 1

            targets = ""
            phonemes_list = []
            notes = [np.empty(shape=(0, 1), dtype=np.short), np.empty(shape=(0, 2))]
            if first_word_to_include - 1 == last_word_to_include + 1: # the word covers the whole window
                # invalid sample, skip
                targets = None
                index = np.random.randint(self.length)
                continue
            if first_word_to_include <= last_word_to_include: # the window covers word[first:last+1]
                # build lyrics target
                lyrics = self.hdf_dataset[str(song_idx)]["lyrics"][first_word_to_include:last_word_to_include+1]
                lyrics_list = [s[0].decode() for s in list(lyrics)]
                targets = " ".join(lyrics_list)
                targets = " ".join(targets.split())

                phonemes = self.hdf_dataset[str(song_idx)]["phonemes"][first_word_to_include:last_word_to_include+1]
                phonemes_list = self.convert_phone_list(phonemes)

            if first_note_to_include <= last_note_to_include:  # the window overlaps or covers note[first:last+1]
                # build melody target
                assert (first_note_to_include <= last_note_to_include)
                notes[0] = self.hdf_dataset[str(song_idx)]["pitches"][first_note_to_include:last_note_to_include + 1]
                notes[1] = self.hdf_dataset[str(song_idx)]["note_times"][first_note_to_include:last_note_to_include + 1,
                           :] - start_pos / self.sr

            if len(targets) > 120 or (len(notes[0]) > 0 and (
                    np.min(notes[0]) < 38 or np.max(notes[0]) > 83)):  # abnormal lyrics length and pitches
                index = np.random.randint(self.length)
                continue

            seq = self.text2seq(targets)
            phone_seq = self.phone2seq(phonemes_list)
            notes[0] = notes[0] - 38

            break

        return audio, targets, seq, phone_seq, notes

    def text2seq(self, text):
        seq = []
        for c in text.lower():
            idx = string.ascii_lowercase.find(c)
            if idx == -1:
                if c == "'":
                    idx = 26
                elif c == " ":
                    idx = 27
                else:
                    continue # remove unknown characters
            seq.append(idx)
        return np.array(seq)

    def phone2seq(self, text):
        seq = []
        for c in text:
            idx = phone2int[c]
            seq.append(idx)
        return np.array(seq)

    def convert_phone_list(self, phonemes):
        ret = []
        for l in phonemes:
            l_decode = [' '] + [s.decode() for s in l if len(s) > 0]
            ret += l_decode
        if len(ret) > 1:
            return ret[1:]
        else:
            return []

    def __len__(self):
        return self.length

class JamendoLyricsDataset(Dataset):
    def __init__(self, sr, hdf_dir, dataset, jamendo_dir, audio_dir, in_memory=False, unit='phone'):
        super(JamendoLyricsDataset, self).__init__()
        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, dataset + ".hdf5")

        self.sr = sr
        self.in_memory = in_memory
        self.unit=unit

        lyrics_dir = os.path.join(jamendo_dir, 'lyrics')
        self.audio_list = [file for file in os.listdir(os.path.join(jamendo_dir, 'mp3')) if file.endswith('.mp3')]

        # create hdf file
        if not os.path.exists(self.hdf_file):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, audio_name in enumerate(tqdm(self.audio_list)):

                    # load audio
                    y, _ = load(os.path.join(audio_dir, audio_name[:-4] + "_vocals.mp3"), sr=self.sr, mono=True)

                    lyrics, words, idx_in_full, idx_line, raw_lines = load_lyrics(os.path.join(lyrics_dir, audio_name[:-4]))
                    lyrics_p, words_p, idx_in_full_p, idx_line_p = gen_phone_gt(words, raw_lines)

                    print(audio_name)
                    annot_num = len(words)
                    line_num = len(idx_line)

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["input_length"] = y.shape[1]
                    grp.attrs["audio_name"] = audio_name[:-4]
                    # print(len(lyrics))

                    grp.create_dataset("lyrics", shape=(1, 1), dtype='S3000', data=np.array([lyrics.encode()]))
                    grp.create_dataset("idx", shape=(annot_num, 2), dtype=np.int, data=idx_in_full)
                    grp.create_dataset("idx_line", shape=(line_num, 2), dtype=np.int, data=idx_line)

                    grp.create_dataset("lyrics_p", shape=(len(lyrics_p), 1), dtype='S2',
                                       data=np.array([l_p.encode() for l_p in lyrics_p]))
                    grp.create_dataset("idx_p", shape=(annot_num, 2), dtype=np.int, data=idx_in_full_p)
                    grp.create_dataset("idx_line_p", shape=(line_num, 2), dtype=np.int, data=idx_line_p)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected.")

        with h5py.File(self.hdf_file, "r") as f:
            self.length = len(f) # number of songs

    def __getitem__(self, index):

        # open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        audio_length = self.hdf_dataset[str(index)].attrs["input_length"]

        # read audio, name, and lyrics
        audio = self.hdf_dataset[str(index)]["inputs"][0, :].astype(np.float32)
        audio_name = self.hdf_dataset[str(index)].attrs["audio_name"]
        if self.unit == 'phone': # load phonemes
            lyrics = self.hdf_dataset[str(index)]["lyrics_p"][:, 0]
            lyrics = [l.decode() for l in lyrics]
            word_idx = self.hdf_dataset[str(index)]["idx_p"]
            line_idx = self.hdf_dataset[str(index)]["idx_line_p"][:]
        else: # load characters
            lyrics = self.hdf_dataset[str(index)]["lyrics"][0, 0].decode()
            word_idx = self.hdf_dataset[str(index)]["idx"]
            line_idx = None

        chunks = [audio]

        # audio, (indices of the first characters/phonemes of the words, * of the lines),
        # (lyrics in characters/phonemes, song names, audio length in samples)
        return chunks, (word_idx, line_idx), (lyrics, audio_name, audio_length)

    def __len__(self):
        return self.length