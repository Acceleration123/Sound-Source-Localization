import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from tqdm import tqdm
from einops import rearrange


class SRP(nn.Module):
    """
    Steering Response Power (SRP) For DOA Estimation Exploiting Uniform Linear Array (ULA)
    Delay and Sum (DS) Beam
    """
    def __init__(
            self,
            d_inter,
            c=343,
            n_fft=512,
            hop_length=256,
            window_length=512,
            fs=16000
    ):
        super(SRP, self).__init__()
        self.d_inter = d_inter
        self.c = c
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length).pow(0.5)
        self.fs = fs

    def multi_channel_stft(self, x):
        """
        input: x: (T (n_samples), C (channels))
        output: X: (C, F, T) complex valued
                f_analog: (F - 1,) except f = 0
        """
        device = x.device
        x = x.unsqueeze(0)  # (1, T, C)
        bs = x.shape[0]
        x = x.transpose(1, 2).reshape(-1, x.size(1))

        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(device),
                       onesided=True, return_complex=True)

        X = torch.view_as_real(X)  # (1, F, T, 2)
        X = X.view(bs, -1, X.shape[1], X.shape[2], 2)  # (1, C, F, T, 2)

        X = torch.complex(X[..., 0], X[..., 1])  # (1, C, F, T)
        f_analog = torch.linspace(0, self.fs / 2, self.n_fft // 2 + 1)

        return X.squeeze(0), f_analog[1:]

    def srp_ula(self, sig_stft, f_analog):
        """
        sig_stft: (C, F, T)
        f_analog: (F - 1,) except f = 0
        """
        array_num = sig_stft.shape[0]
        theta = torch.linspace(-90, 90, 361)
        theta = torch.deg2rad(theta)
        theta_sin = torch.sin(theta)

        sig_stft = rearrange(sig_stft, 'c f t -> t f c')  # (T, F, C)
        power_map = torch.zeros(len(f_analog), len(theta))  # (F, angle)

        for i in tqdm(range(len(f_analog))):
            f = f_analog[i]
            for j in range(len(theta)):
                phase = (2 * torch.pi * f * theta_sin[j] * self.d_inter *
                         torch.linspace(0, array_num - 1, array_num) / self.c)  # (C,)
                steer_vec = torch.exp(-1j * phase)  # (C,)
                sig_mic = sig_stft[:, i + 1, :]  # (T, C)
                response_complex = torch.matmul(sig_mic.unsqueeze(1), steer_vec.unsqueeze(1))
                response_norm = torch.mean(torch.abs(response_complex).squeeze(1))
                power_map[i, j] = response_norm ** 2

        power_map_mean = torch.mean(power_map, dim=0)  # (angle,)
        doa_order = torch.argmax(power_map_mean)

        return doa_order * 0.5 - 90, power_map_mean

    def forward(self, x):
        """
        input: x: (T (n_samples), C (channels))
        output: doa_estimation: (1,)
        """
        sig_stft, f_analog = self.multi_channel_stft(x)
        doa_est, power_map = self.srp_ula(sig_stft, f_analog)

        return doa_est, power_map


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speech, fs = sf.read('00001.wav', dtype='float32')
    rir = np.load('rir/test/direct/0017_rt60=0.86_d=0.020_r1=1.00_phi1=64.94.npy')

    sensor_1 = signal.fftconvolve(speech, rir[0][0], mode='same')
    sensor_2 = signal.fftconvolve(speech, rir[1][0], mode='same')
    sensor_3 = signal.fftconvolve(speech, rir[2][0], mode='same')
    sensor_4 = signal.fftconvolve(speech, rir[3][0], mode='same')

    sig = np.stack([sensor_1, sensor_2, sensor_3, sensor_4], axis=1)  # (T, 4)
    sig = torch.tensor(sig).to(device)

    srp_func = SRP(d_inter=0.02)
    doa_est, power_map = srp_func(sig)
    print(f"DOA Estimation: {doa_est:.2f}°")

    power_map = power_map.cpu().detach().numpy()
    theta_deg = np.linspace(-90, 90, power_map.shape[0])
    plt.plot(theta_deg, power_map, linestyle='--', color='blue', label='Power Map')

    max_index = np.argmax(power_map)
    plt.axvline(x=theta_deg[max_index], color='red', linestyle='--', linewidth=2, label='DOA Estimation')
    plt.scatter(theta_deg[max_index], power_map[max_index], color='red', s=100, zorder=3, edgecolors='black')
    plt.annotate(
        f'DOA Estimation: {theta_deg[max_index]:.1f}°',
        (theta_deg[max_index], power_map[max_index]),
        xytext=(20, 20), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='red')
    )

    plt.xlabel('Elevation Angle (degrees)')
    plt.ylabel('Power')
    plt.title('SRP Power Map')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlim(-90, 90)  # 显式设置x轴范围
    plt.show()

