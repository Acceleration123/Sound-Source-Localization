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
            nn.ELU()
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
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.out_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.out_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU()
        )

        self.out_conv_main = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )

        self.out_conv_secondary = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, F, T) 增加通道维度
        # Encoder
        x_en1 = self.en_conv1(x)
        x_en2 = self.en_conv2(x_en1)
        x_en3 = self.en_conv3(x_en2)
        x_en4 = self.en_conv4(x_en3)

        # Decoder
        x_de1 = self.en_conv5(x_en4)
        # Skip Connection
        x_de1 = torch.cat([x_de1, x_en4], dim=1)
        x_de2 = self.de_conv1(x_de1)
        x_de2 = torch.cat([x_de2, x_en3], dim=1)
        x_de3 = self.de_conv2(x_de2)
        x_de3 = torch.cat([x_de3, x_en2], dim=1)
        x_de4 = self.de_conv3(x_de3)
        x_de4 = torch.cat([x_de4, x_en1], dim=1)  # (Batch_size, 192, 256, 256)

        # Output Block
        out_main1 = out_secondary1 = self.out_conv1(x_de4)
        out_main2 = self.out_conv2(out_main1)
        out_main3 = self.out_conv2(out_main2)
        out_main = self.out_conv_main(out_main3)  # (B, 1, F, T)

        out_secondary1 = torch.cat([out_secondary1, out_main1], dim=1)
        out_secondary2 = self.out_conv3(out_secondary1)
        out_secondary2 = torch.cat([out_secondary2, out_main2], dim=1)
        out_secondary3 = self.out_conv3(out_secondary2)
        out_secondary3 = torch.cat([out_secondary3, out_main3], dim=1)
        out_secondary = self.out_conv_secondary(out_secondary3)  # (B, 1, F, T)

        out_main = out_main.squeeze(1)  # (B, F, T)
        out_secondary = out_secondary.squeeze(1)  # (B, F, T)

        return out_main, out_secondary















