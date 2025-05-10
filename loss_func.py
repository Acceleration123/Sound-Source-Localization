import torch
import torch.nn as nn


class loss_func(nn.Module):
    def __init__(self):
        super(loss_func, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, irms_pred, irmd_pred, irms_true, irmd_true):
        """
        input: (B, F, T)
        F=256, T=256 in this implementation
        """
        loss = 0.5 * self.mse(irms_pred, irms_true) + 0.5 * self.mse(irmd_pred, irmd_true)

        return loss


if __name__ == '__main__':
    x1 = torch.randn(2, 256, 256)
    x2 = torch.randn(2, 256, 256)
    y1 = torch.randn(2, 256, 256)
    y2 = torch.randn(2, 256, 256)

    loss_func = loss_func()
    loss = loss_func(x1, x2, y1, y2)
    print(loss)


