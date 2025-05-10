import torch
from scipy import signal
from torch.utils import data
import soundfile as sf
import numpy as np
import librosa
import random
import pandas as pd
from tqdm import tqdm

TRAIN_DATABASE_CLEAN = 'D:\\Users\\14979\\Desktop\\DNS5_16k\\train_clean'
TRAIN_NOISE_DATABASE = 'D:\\Users\\Database\\train_noise_data_new.csv'
RIR_DIRECT_DATABASE = 'rir/train/direct'
RIR_REVERB_DATABASE = 'rir/train/reverb'


def get_irm(direct, reverb, noisy):
    """
    STFT representation
    direct: (F, T)
    reverb: (F, T)
    noisy: (F, T)
    F = 256, T = 256 in this implementation
    """
    eps = 1e-8
    direct_energy = torch.abs(direct)  # (F, T)
    reverb_energy = torch.abs(reverb)  # (F, T)
    noisy_energy = torch.abs(noisy)  # (F, T)

    # IRM_Direct
    irm_direct = direct_energy / (noisy_energy + eps)   # (F, T)

    # IRM_Speech
    irm_speech = (direct_energy + reverb_energy) / (noisy_energy + eps)   # (F, T)
    irm_speech[irm_speech < 0.5] = eps  # use boolean indexing instead of for loop

    return irm_direct, irm_speech


def stft_encoder(x, n_fft=512, hop_length=256, window=torch.hann_window(512).pow(0.5)):
    """
    x: (T,)  time domain
    """
    device = x.device
    x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=window.to(device), return_complex=True)  # (F, T)

    return x_stft  # (F, T)


class Train_Dataset(data.Dataset):
    def __init__(
            self,
            fs=16000,
            length_in_seconds=5,
            num_data_tot=60000,
            num_data_per_epoch=10000,
            random_start_point=False,
            eps=1e-8
    ):
        super(Train_Dataset, self).__init__()
        self.train_noisy_database = sorted(librosa.util.find_files(TRAIN_DATABASE_CLEAN, ext='wav'))[:num_data_tot]
        self.rir_direct_database = sorted(librosa.util.find_files(RIR_DIRECT_DATABASE, ext='npy'))
        self.rir_reverb_database = sorted(librosa.util.find_files(RIR_REVERB_DATABASE, ext='npy'))
        self.noise_database = sorted(pd.read_csv(TRAIN_NOISE_DATABASE)['file_dir'].tolist())[:num_data_tot]

        self.L = int(fs * length_in_seconds)
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.random_start_point = random_start_point
        self.eps = eps

    def __len__(self):
        return self.num_data_per_epoch

    def __getitem__(self, idx):
        # 随机抽取
        clean_list = random.sample(self.train_noisy_database, self.num_data_per_epoch)
        noise_list = random.sample(self.noise_database, self.num_data_per_epoch)

        # 裁剪
        if self.random_start_point:
            Begin_S = int(np.random.uniform(0, 10 - self.length_in_seconds)) * self.fs
            clean, _ = sf.read(clean_list[idx], dtype='float32', start=Begin_S, stop=Begin_S + self.L)
            noise, _ = sf.read(noise_list[idx], dtype='float32', start=Begin_S, stop=Begin_S + self.L)
        else:
            clean, _ = sf.read(clean_list[idx], dtype='float32', start=0, stop=self.L)
            noise, _ = sf.read(noise_list[idx], dtype='float32', start=0, stop=self.L)

        index = np.random.randint(0, len(self.rir_direct_database) - 1)

        rir_direct = np.load(self.rir_direct_database[index])[0][0]
        rir_reverb = np.load(self.rir_reverb_database[index])[0][0]

        # 对齐
        T = len(clean)
        direct = signal.fftconvolve(clean, rir_direct, mode='full')[:T]
        mix = signal.fftconvolve(clean, rir_reverb, mode='full')[:T]
        reverb = mix - direct
        snr = np.random.uniform(-5, 20)
        alpha = np.sqrt(np.var(mix) * (10 ** (-snr / 10)) / (np.var(noise + self.eps)))
        noisy = mix + alpha * noise

        direct = torch.tensor(direct)  # (T,)
        reverb = torch.tensor(reverb)  # (T,)
        noisy = torch.tensor(noisy)    # (T,)

        direct_stft = stft_encoder(direct)[0:256, 0:256]  # (F, T)
        reverb_stft = stft_encoder(reverb)[0:256, 0:256]  # (F, T)
        noisy_stft = stft_encoder(noisy)[0:256, 0:256]    # (F, T)

        irm_direct, irm_speech = get_irm(direct_stft, reverb_stft, noisy_stft)

        return noisy, irm_direct, irm_speech


class Valid_Dataset(data.Dataset):
    def __init__(self):
        super(Valid_Dataset, self).__init__()
        self.direct_path = os.path.join(VALID_DATASET_PATH, 'direct')
        self.reverb_path = os.path.join(VALID_DATASET_PATH, 'reverb')
        self.noisy_path = os.path.join(VALID_DATASET_PATH, 'noisy')

        self.direct_list = sorted(librosa.util.find_files(self.direct_path, ext='wav'))
        self.reverb_list = sorted(librosa.util.find_files(self.reverb_path, ext='wav'))
        self.noisy_list = sorted(librosa.util.find_files(self.noisy_path, ext='wav'))

    def __len__(self):
        return len(self.direct_list)

    def __getitem__(self, idx):
        direct, _ = sf.read(self.direct_list[idx], dtype='float32')
        reverb, _ = sf.read(self.reverb_list[idx], dtype='float32')
        noisy, _ = sf.read(self.noisy_list[idx], dtype='float32')

        direct = torch.tensor(direct)  # (T,)
        reverb = torch.tensor(reverb)  # (T,)
        noisy = torch.tensor(noisy)    # (T,)

        direct_stft = stft_encoder(direct)[0:256, 0:256]  # (F, T)
        reverb_stft = stft_encoder(reverb)[0:256, 0:256]  # (F, T)
        noisy_stft = stft_encoder(noisy)[0:256, 0:256]    # (F, T)

        irm_direct, irm_speech = get_irm(direct_stft, reverb_stft, noisy_stft)  # (F, T)

        return noisy, irm_direct, irm_speech


if __name__ == '__main__':
    # test get_irm function
    clean, fs = sf.read('00001.wav', dtype='float32')
    rir_direct = np.load('rir/test/direct/0000_rt60=0.81_d=0.020_r1=0.50_phi1=19.52.npy')[0][0]
    rir_reverb = np.load('rir/test/reverb/0000_rt60=0.81_d=0.020_r1=0.50_phi1=19.52.npy')[0][0]

    T = len(clean)
    direct = signal.fftconvolve(clean, rir_direct, mode='full')[:T]
    mix = signal.fftconvolve(clean, rir_reverb, mode='full')[:T]
    reverb = mix - direct
    noise = np.random.normal(0, 1, len(clean))
    noisy = mix + noise

    direct = torch.tensor(direct)  # (T,)
    reverb = torch.tensor(reverb)  # (T,)
    noisy = torch.tensor(noisy)    # (T,)

    direct_stft = stft_encoder(direct)[0:256, 0:256]  # (F, T)
    reverb_stft = stft_encoder(reverb)[0:256, 0:256]  # (F, T)
    noisy_stft = stft_encoder(noisy)[0:256, 0:256]    # (F, T)

    irm_direct, irm_speech = get_irm(direct_stft, reverb_stft, noisy_stft)
    print(irm_direct.shape, irm_speech.shape)

    # test Train_Dataset
    train_dataset = Train_Dataset(num_data_per_epoch=10)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (noisy, irm_direct, irm_speech) in enumerate(tqdm(train_loader)):
        print(noisy.shape, irm_direct.shape, irm_speech.shape)
        
    # test Valid_Dataset
    valid_dataset = Valid_Dataset()
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (noisy, irm_direct, irm_speech) in enumerate(tqdm(valid_loader)):
        print(noisy.shape, irm_direct.shape, irm_speech.shape)







