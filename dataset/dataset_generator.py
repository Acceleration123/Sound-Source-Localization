import numpy as np
import soundfile as sf
import librosa
import random
from scipy import signal
import os
import argparse
import pandas as pd
from tqdm import tqdm


class Dataset_Generator:
    def __init__(self, args, eps=1e-8):
        self.speech_dir = args.speech_dir
        self.noise_csv = args.noise_csv_path
        self.rir_dir = args.rir_dir
        self.save_dir = args.save_dir
        self.eps = eps

    def generator(self):
        speech_list = sorted(librosa.util.find_files(self.speech_dir, ext='wav'))

        # 噪声随机抽取
        noise_list_tot = sorted(pd.read_csv(self.noise_csv)['file_dir'].tolist())
        noise_list = random.sample(noise_list_tot, len(speech_list))

        rir_direct_list_tot = sorted(librosa.util.find_files(self.rir_dir.replace('reverb', 'direct'), ext='npy'))
        rir_reverb_list_tot = sorted(librosa.util.find_files(self.rir_dir, ext='npy'))

        rir_direct_list = random.sample(rir_direct_list_tot, len(speech_list))
        rir_reverb_list = random.sample(rir_reverb_list_tot, len(speech_list))

        direct_dir = os.path.join(self.save_dir, 'direct')
        reverb_dir = os.path.join(self.save_dir, 'reverb')
        noisy_dir = os.path.join(self.save_dir, 'noisy')

        os.makedirs(direct_dir, exist_ok=True)
        os.makedirs(reverb_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        file_length = 4
        for i in tqdm(range(len(speech_list))):
            speech, _ = sf.read(speech_list[i], dtype='float32')
            T = len(speech) // 2  # (80000,)

            noise, _ = sf.read(noise_list[i], dtype='float32')
            noise = noise[:T]

            rir_direct = np.load(rir_direct_list[i])[0][0]
            rir_reverb = np.load(rir_reverb_list[i])[0][0]
            snr = random.uniform(-5, 20)

            direct = signal.fftconvolve(speech, rir_direct, mode='full')[:T]
            mix = signal.fftconvolve(speech, rir_reverb, mode='full')[:T]
            reverb = mix - direct

            alpha = np.sqrt(np.var(mix) * (10 ** (-snr / 10)) / (np.var(noise + self.eps)))
            noisy = mix + alpha * noise

            sf.write(os.path.join(direct_dir, f"{i:0{file_length}d}.wav"), direct, 16000)
            sf.write(os.path.join(reverb_dir, f"{i:0{file_length}d}.wav"), reverb, 16000)
            sf.write(os.path.join(noisy_dir, f"{i:0{file_length}d}.wav"), noisy, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-speech_dir', '--speech_dir', type=str,
                        default="D:\\Users\\14979\\Desktop\\DNS5_16k\\dev_clean")

    parser.add_argument('-noise_csv', '--noise_csv_path', type=str,
                        default="D:\\Users\\Database\\valid_noise_data_new.csv")

    parser.add_argument('-rir_dir', '--rir_dir', type=str,
                        default="C:\\Users\\14979\\Desktop\\NN-Zoo\\U-Net-for-SSL\\rir\\validation\\reverb",
                        help='rir in the reverberant room ')

    parser.add_argument('-save_dir', '--save_dir', type=str,
                        default="D:\\Users\\SSL\\valid")

    args = parser.parse_args()

    dataset_generator = Dataset_Generator(args)
    dataset_generator.generator()
