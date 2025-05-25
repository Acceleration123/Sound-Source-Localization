import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from scipy import signal


class TDOA(nn.Module):
    """
    Time Delay of Arrival (TDOA) For DOA Estimation Exploiting Two Microphones
    In this implementation, GCC-PHAT is used to estimate DOA.
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
        super(TDOA, self).__init__()
        self.d_inter = d_inter
        self.c = c
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length)
        self.fs = fs

    def multi_channel_stft(self, x):
        """
        input: x: (T (n_samples), C (channels))
        output: X: (C, F, T) complex valued
        Note that onesided=False is used to get the full complex STFT representation i.e. F = n_fft
        """
        device = x.device
        x = x.unsqueeze(0)  # (1, T, C)
        bs = x.shape[0]
        x = x.transpose(1, 2).reshape(-1, x.size(1))

        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(device),
                       onesided=False, return_complex=True)

        X = torch.view_as_real(X)  # (1, F, T, 2)
        X = X.view(bs, -1, X.shape[1], X.shape[2], 2)  # (1, C, F, T, 2)

        X = torch.complex(X[..., 0], X[..., 1])  # (1, C, F, T)

        return X.squeeze(0)

    def inverse_dft(self, sig_dft):
        """
        sig_dft: (F, )
        output: sig: (T, )
        where T = F = n_fft
        """
        order = torch.linspace(0, self.n_fft - 1, self.n_fft, dtype=torch.float32).unsqueeze(1)
        dft_mat = torch.exp(1j * 2 * torch.pi * order.mul(order.transpose(0, 1)) / self.n_fft) / self.n_fft   # (F, F)
        sig_dft = sig_dft.unsqueeze(1)  # (F, 1)
        sig = torch.matmul(dft_mat, sig_dft)   # (F, 1)

        return sig.squeeze(1)   # (T,)

    def get_gcc(self, sig_stft, eps=1e-8):
        """
        input: sig_stft: (2, F)  STFT representation between adjacent sensors at a fixed frame
        output: r: cross-correlation of time delay tao
        """
        sensor1_stft, sensor2_stft = sig_stft[0], sig_stft[1]   # (F,)
        csp = sensor1_stft * (sensor2_stft.conj())   # (F,)
        csp = csp / (torch.norm(csp) + eps)   # (F,)
        r_tao = self.inverse_dft(csp)   # (T,)

        return r_tao

    def get_doa(self, sig_stft):
        """
        input: x: (T (n_samples), C (channels))
        output: doa_est: DOA in degree
        C = 2 in this implementation
        """
        r_tao = self.get_gcc(sig_stft)  # (T,)

        _, max_tao = torch.topk(torch.real(r_tao), k=3)
        sin_doa = torch.mean(max_tao.float()) * self.c / (self.d_inter * self.fs)
        doa_est = torch.asin(sin_doa).rad2deg()

        return doa_est

    def forward(self, x):
        X = self.multi_channel_stft(x)[..., 126]  # (C = 2, F, T)
        doa_est = self.get_doa(X)

        if torch.isnan(doa_est):
            X = torch.flip(X, dims=[0])
            doa_est = -self.get_doa(X)

        return doa_est


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speech, _ = sf.read('conventional algorithms/00001.wav', dtype='float32')
    rir = np.load('0001_rt60=0.81_d=0.400_r1=0.50_phi1=-41.92.npy')

    T = len(speech)
    sensor_1 = signal.fftconvolve(speech, rir[1][0], mode='full')[:T]  # (T,)
    sensor_2 = signal.fftconvolve(speech, rir[2][0], mode='full')[:T]  # (T,)

    sf.write('sensor_1.wav', sensor_1, samplerate=16000)
    sf.write('sensor_2.wav', sensor_2, samplerate=16000)

    sig = np.stack([sensor_1, sensor_2], axis=-1)  # (T, 2)
    sig = torch.tensor(sig).to(device)  # (T, 2)

    tdoa_func = TDOA(d_inter=0.4)
    doa = tdoa_func(sig)
    print(doa)
