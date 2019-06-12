import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from torchvision import transforms


LABELS = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum',
          'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz',
          'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking', 'Chink_and_clink',
          'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd',
          'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans', 'Drawer_open_or_close',
          'Drip', 'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
          'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss',
          'Keys_jangling', 'Knock', 'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone',
          'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing',
          'Raindrop', 'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard',
          'Slam', 'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
          'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet',
          'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


class KFSDataset(Dataset):
    def __init__(self, data, transform):
        super(KFSDataset, self).__init__()
        self.data = data
        self.n_labels = len(LABELS)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, labels = self.data[item]
        wav, sr = torchaudio.load(path)
        # print('raw', wav.size())
        logmel = self.transform(wav)
        # print('logmel', logmel.size())
        labels = np.eye(self.n_labels)[labels].sum(axis=0).astype(np.float32)
        return dict(
            logmel=logmel,
            labels=labels,
            fname=os.path.basename(path),
        )


def load_data(dataroot, kind=None):
    assert kind in {'train_curated', 'train_noisy'}
    csv_path = os.path.join(dataroot, kind + '.csv')
    df = pd.read_csv(csv_path, sep=',')
    data = []
    print('loading data')
    for fname, s in tqdm(zip(df[u'fname'].values, df[u'labels'].values)):
        path = os.path.join(dataroot, kind, fname)
        labels = [label2id[k] for k in s.split(',')]
        data.append((path, labels))
    return data


def build_dataset(dataroot, transform):
    whole_data = []
    for ds in ['train_noisy', 'train_curated']:
        whole_data += load_data(dataroot, kind=ds)
    idx = np.arange(len(whole_data))

    np.random.seed(2019)
    np.random.shuffle(idx)
    n = int(0.33 * len(whole_data))
    train_idx, val_idx = idx[n:], idx[:n]
    trainds = KFSDataset([whole_data[_] for _ in train_idx], transform)
    evalds = KFSDataset([whole_data[_] for _ in val_idx], transform)
    return trainds, evalds


def build_preprocessing(model, ref_len=665100, batch_size=32, low_db=-10.0, high_db=120.0, mean_db=1.0, std_db=2.5):
    if model == 'cgrnn':
        n_fft, hop, n_mels = 5296, 2648, 64
    else:
        n_fft, hop, n_mels = 1764, 220, 256
    basic_transform = [
        torchaudio.transforms.MelSpectrogram(sr=44100, n_fft=n_fft, hop=hop, n_mels=n_mels),
        torchaudio.transforms.SpectrogramToDB(),
        lambda _: torch.clamp(_, low_db, high_db),
        lambda _: (_ - mean_db) / std_db,
    ]

    if ref_len:
        ref_len = int(ref_len)
        preprocessing = [torchaudio.transforms.PadTrim(ref_len)] + basic_transform
    else:
        assert batch_size == 1
        preprocessing = basic_transform

    return transforms.Compose(preprocessing)
