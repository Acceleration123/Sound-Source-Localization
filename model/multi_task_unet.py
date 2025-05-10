import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


# U-Net 架构
class UNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(UNet, self).__init__()

        self.dropout = dropout

        # Encoder
        # 注意保证每一个卷积层的输出和输入的大小一致
        self.en_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.en_conv2 = nn.Sequential(
            # 最大池化层 输出的大小为原来的1/2
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.en_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU()
        )

        self.en_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ELU()
        )

        self.en_conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(512, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(768),
            nn.ELU(),
            nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(768),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(768, 768, kernel_size=(2, 2), stride=(2, 2))
        )

        # Decoder
        # 注意保证每一个卷积层的输出和输入的大小一致
        # 注意每一次上采样之后输出的大小变为原来的2倍
        # 以及要注意U-Net的跳跃链接
        self.de_conv1 = nn.Sequential(
            nn.Conv2d(1280, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(768),
            nn.ELU(),
            nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
        )

        self.de_conv2 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
        )

        self.de_conv3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        )

        # Output Block
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.main_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.main_conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.main_conv4 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )

        self.auxiliary_conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.auxiliary_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.auxiliary_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.auxiliary_conv4 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, T) time domain
        """
        device = x.device
        x = torch.stft(x, n_fft=512, hop_length=256, window=torch.hann_window(512).pow(0.5).to(device),
                       return_complex=True)[:, 0:256, 0:256]  # (B, F, T)
        x = torch.abs(x).log10()  # (B, F, T)

        x = x.unsqueeze(1)  # (B, 1, F, T) 增加通道维度
        # Encoder
        x_en1 = self.en_conv1(x)  # (B, 64, 256, 256)
        x_en2 = self.en_conv2(x_en1)  # (B, 128, 128, 128)
        x_en3 = self.en_conv3(x_en2)  # (B, 256, 64, 64)
        x_en4 = self.en_conv4(x_en3)  # (B, 512, 32, 32)

        # Decoder
        x_de1 = self.en_conv5(x_en4)  # (B, 768, 32, 32)
        # Skip Connection
        x_de1 = torch.cat([x_de1, x_en4], dim=1)
        x_de2 = self.de_conv1(x_de1)  # (B, 512, 64, 64)
        x_de2 = torch.cat([x_de2, x_en3], dim=1)
        x_de3 = self.de_conv2(x_de2)  # (B, 256, 128, 128)
        x_de3 = torch.cat([x_de3, x_en2], dim=1)
        x_de4 = self.de_conv3(x_de3)  # (B, 128, 256, 256)
        x_de4 = torch.cat([x_de4, x_en1], dim=1)  # (B, 192, 256, 256)

        # Output Block
        # Main Task
        x_main_1 = self.main_conv1(x_de4)  # (B, 128, 256, 256)
        x_main_2 = self.main_conv2(x_main_1)  # (B, 128, 256, 256)
        x_main_3 = self.main_conv3(x_main_2)  # (B, 128, 256, 256)
        out_main = self.main_conv4(x_main_3)  # (B, 1, 256, 256)

        # Auxiliary Task
        x_aux_1 = self.auxiliary_conv1(x_de4)  # (B, 128, 256, 256)
        x_aux_2 = torch.cat([x_aux_1, x_main_1], dim=1)  # (B, 256, 256, 256)
        x_aux_2 = self.auxiliary_conv2(x_aux_2)  # (B, 128, 256, 256)
        x_aux_3 = torch.cat([x_aux_2, x_main_2], dim=1)  # (B, 256, 256, 256)
        x_aux_3 = self.auxiliary_conv3(x_aux_3)  # (B, 128, 256, 256)
        out_aux = torch.cat([x_aux_3, x_main_3], dim=1)  # (B, 256, 256, 256)
        out_aux = self.auxiliary_conv4(out_aux)  # (B, 1, 256, 256)

        return out_main, out_aux


if __name__ == '__main__':
    model = UNet().eval()
    macs, params = get_model_complexity_info(model, (80000,), as_strings=True, print_per_layer_stat=True)
    print(macs, params)















