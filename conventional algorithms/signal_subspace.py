import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm
from matplotlib import pyplot as plt


def propagation(source, rir, t_len):
    x = []
    for i in range(len(rir)):
        x.append(signal.fftconvolve(source, rir[i], mode='full')[:t_len])

    return np.stack(x, axis=-1)  # (T, C)


def multi_channel_stft(x, n_fft=512, hop_length=256, window=torch.hann_window(512).pow(0.5)):
    """
    input: x: (T (n_samples), C (channels))
    output: X: (C, F, T) complex valued
    """
    device = x.device
    x = x.unsqueeze(0)  # (1, T, C)
    bs = x.shape[0]
    x = x.transpose(1, 2).reshape(-1, x.size(1))

    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=window.to(device),
                   onesided=True, return_complex=True)

    X = torch.view_as_real(X)  # (1, F, T, 2)
    X = X.view(bs, -1, X.shape[1], X.shape[2], 2)  # (1, C, F, T, 2)

    X = torch.complex(X[..., 0], X[..., 1])  # (1, C, F, T)

    return X.squeeze(0)


def get_spt_mat(X, est_num, t_start):
    """
    input:
    X: (C, F, T)  STFT representation
    est_num: number of TF bins used to estimate spatial covariance matrix
    t_start: starting frame index of the TF bins used to estimate spatial covariance matrix,
    and it can be obtained using VAD
    output:
    spt_mat: (F, C, C) spatial covariance matrix
    """
    device = X.device
    M, F, _ = X.shape
    X = X.permute(1, 0, 2)   # (F, C, T)
    spt_mat = torch.zeros(F, M, M, dtype=torch.complex64).to(device)
    for i in range(est_num):
        X_vec = X[:, :, t_start + i].unsqueeze(-1)  # (F, C, 1)
        spt_mat += torch.matmul(X_vec, (torch.conj(X_vec).transpose(1, 2)))  # (F, C, C)

    return spt_mat / est_num  # (F, C, C)


class MUSIC(nn.Module):
    def __init__(
            self,
            d_inter,
            source_num=2,
            mic_num=4,
            c=343,
            n_fft=512,
            fs=16000,
            f_low=400,
            f_high=1000,
            resolution=0.5
    ):
        """
        Implementation of MUSIC algorithm for sound source localization.
        """
        super(MUSIC, self).__init__()
        self.d_inter = d_inter
        self.source_num = source_num
        self.mic_num = mic_num
        self.c = c
        self.n_fft = n_fft
        self.fs = fs
        self.f_low = f_low
        self.f_high = f_high
        self.resolution = resolution

    def spatial_nyquist_check(self):
        if self.d_inter > self.c / (2 * self.f_high):
            raise ValueError("Spatial Nyquist theorem not satisfied!")

    def source_num_check(self):
        if self.source_num > self.mic_num:
            raise ValueError("Number of sources should be less than or equal to number of microphones!")

    def get_spt_spec(self, spt_mat, f_analog, eps=1e-8):
        """
        spt_mat: (F, C, C)  spatial covariance matrix
        f_analog: (F,)
        """
        M = spt_mat.shape[-1]
        theta = torch.linspace(-90, 90, int(180 / self.resolution + 1))
        theta = torch.deg2rad(theta)
        theta_sin = torch.sin(theta)
        eig_value, eig_mat = torch.linalg.eig(spt_mat)   # (F, C) & (F, C, C)
        eig_mat = eig_mat[:, :, self.source_num:]  # (F, C, source_num)

        spt_spec = torch.zeros(len(f_analog), len(theta))  # (F, angle)
        for i in tqdm(range(len(theta))):
            phase = (2 * torch.pi * theta_sin[i] * self.d_inter *
                     (f_analog.unsqueeze(-1)).mul(torch.linspace(0, M - 1, M).unsqueeze(0)) / self.c)  # (F, C)
            steer_vec = torch.exp(-1j * phase)  # (F, C)
            middle_vec = torch.matmul(steer_vec.unsqueeze(1), eig_mat).squeeze(1)  # (F, source_num)
            value = torch.abs(middle_vec).pow(2)
            spt_spec[:, i] = 1 / (torch.sum(value, dim=1, keepdim=False) + eps)

        spt_spec = torch.mean(spt_spec, dim=0, keepdim=False)  # (angle,)

        return spt_spec   # (angle,)

    def forward(self, x):
        """
        input: x: (T (n_samples), C (channels))
        output: spatial spectrum: (angle,)
        """
        self.spatial_nyquist_check()
        self.source_num_check()

        X = multi_channel_stft(x)  # (C, F, T)
        f_analog = torch.linspace(0, self.fs / 2, self.n_fft // 2 + 1)

        f_use_idx = (f_analog >= self.f_low) & (f_analog < self.f_high)
        f_analog = f_analog[f_use_idx]
        X = X[:, f_use_idx, :]  # (C, F, T)

        spt_mat = get_spt_mat(X, 4, 120)  # (F, C, C)
        spt_spec = self.get_spt_spec(spt_mat,  f_analog)

        return spt_spec  # (angle,)


class ESPRIT(nn.Module):
    def __init__(
            self,
            d_inter,
            source_num=2,
            mic_num=4,
            c=343,
            n_fft=512,
            fs=16000,
            f_low=400,
            f_high=1000,
    ):
        """
        # Implementation of ESPRIT algorithm for sound source localization.
        """
        super(ESPRIT, self).__init__()
        self.d_inter = d_inter
        self.source_num = source_num
        self.mic_num = mic_num
        self.c = c
        self.n_fft = n_fft
        self.fs = fs
        self.f_low = f_low
        self.f_high = f_high

    def spatial_nyquist_check(self):
        if self.d_inter > self.c / (2 * self.f_high):
            raise ValueError("Spatial Nyquist theorem not satisfied!")

    def source_num_check(self):
        if self.source_num > self.mic_num:
            raise ValueError("Number of sources should be less than or equal to number of microphones!")

    def esprit(self, spt_mat, f_analog):
        """
        spt_mat: (F, C, C)  spatial covariance matrix
        f_analog: (F,)
        """
        angle_list = []
        for i in tqdm(range(len(f_analog))):
            eig_value, eig_mat = torch.linalg.eig(spt_mat[i])  # (C,) & (C, C)
            sig_mat = eig_mat[:, :self.source_num]  # (C, source_num)
            sig_mat_0, sig_mat_1 = sig_mat[:-1], sig_mat[1:]  # (C - 1, source_num) & (C - 1, source_num)

            phi_mat = torch.matmul(torch.linalg.pinv(sig_mat_0), sig_mat_1)  # (source_num, source_num)
            steer_vec, _ = torch.linalg.eig(phi_mat)  # (source_num,)

            phase = torch.atan2(steer_vec.imag, steer_vec.real)  # (source_num,)
            angle = torch.asin(phase * self.c / (2 * torch.pi * f_analog[i] * self.d_inter)).rad2deg()  # (source_num,)
            angle_list.append(angle)

        angle_list = torch.cat(angle_list, dim=0)  # (F, source_num)

        return torch.mean(angle_list, dim=0, keepdim=False)  # (source_num,)

    def forward(self, x):
        """
        input: x: (T (n_samples), C (channels))
        output: doa_estimation: (1,)
        """
        self.spatial_nyquist_check()
        self.source_num_check()

        X = multi_channel_stft(x)  # (C, F, T)
        f_analog = torch.linspace(0, self.fs / 2, self.n_fft // 2 + 1)

        f_use_idx = (f_analog >= self.f_low) & (f_analog < self.f_high)
        f_analog = f_analog[f_use_idx]
        X = X[:, f_use_idx, :]  # (C, F, T)

        spt_mat = get_spt_mat(X, 4, 120)  # (F, C, C)
        doa_est = self.esprit(spt_mat, f_analog)

        return doa_est  # (F,)


if __name__ == '__main__':
    source0, fs = sf.read('./conventional algorithms/00000.wav', dtype='float32')
    source1, _ = sf.read('./conventional algorithms/00001.wav', dtype='float32')
    T = min(len(source0), len(source1))

    rir = np.load('rir/test/direct/0018_rt60=0.20_d=0.020_r1=0.50_phi1=44.55.npy')

    sample0 = propagation(source0, rir[:, 0], T)  # (T, C)
    sample0 = torch.tensor(sample0)

    music = MUSIC(d_inter=0.02, source_num=1, mic_num=4)
    esprit = ESPRIT(d_inter=0.02, source_num=1, mic_num=4)
    spat_spec = music(sample0)
    doa_est = esprit(sample0)
    print(f'DOA estimation: {doa_est.cpu().detach().numpy()}')

    power_map = spat_spec.cpu().detach().numpy()
    theta_deg = np.linspace(-90, 90, power_map.shape[0])
    plt.plot(theta_deg, power_map, linestyle='--', color='blue', label='Power Map')

    max_index = np.argmax(power_map)
    plt.axvline(x=theta_deg[max_index], color='red', linestyle='--', linewidth=2, label='DOA Estimation')
    plt.scatter(theta_deg[max_index], power_map[max_index], color='red', s=100, zorder=3, edgecolors='black')
    plt.annotate(
        f'DOA Estimation: {theta_deg[max_index]:.2f}°',
        (theta_deg[max_index], power_map[max_index]),
        xytext=(20, 20), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='red')
    )

    plt.xlabel('Elevation Angle (degrees)')
    plt.ylabel('Power')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlim(-90, 90)  # 显式设置x轴范围
    plt.show()

