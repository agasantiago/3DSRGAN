import torch
from torch import nn


from Vgg3D import Vgg3D


class VggLoss(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._vgg_loss = Vgg3D(in_channels=in_channels).eval()
        self._mse_loss = nn.MSELoss()

        for param in self._vgg_loss.parameters():
            param.requires_grad = False

    def forward(self, X, y):
        """
        Calcula a perda "perceptual":
        :param X: Input
        :param y: Target
        :return: Perda considerando "perceptual loss" e "mse"
        """
        X_loss = self._vgg_loss(X)
        y_loss = self._vgg_loss(y)
        loss = self._mse_loss(y_loss, X_loss)
        return loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.zeros((1, 1, 16, 16, 16)).to(device)
    y = torch.ones((1, 1, 16, 16, 16)).to(device)
    vgg3d_loss = VggLoss(in_channels=1).to(device)
    loss = vgg3d_loss(X, y)
    print(loss.item())
