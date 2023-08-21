import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=False,
        use_bn=False,
        **kwargs,
    ):
        """
        ConvBlocks: Cria o bloco para operações de Convolução 3D;
        :param in_channels: Quantidade de canais de entrada (RGB ou GRAYSCALE)
        :param out_channels: Quantidade de canais de saída;
        :param discriminator: True se utilizado na classe Discriminator.
        :param use_act: True se necessário função de ativação (LeakyReLU);
        :param use_bn: True se necessário BacthNorm3d;
        :param kwargs: argumentos para a rede Conv3d (kernel_size, padding, stride, etc);
        """
        super().__init__()
        self._cnn = nn.Conv3d(in_channels, out_channels, **kwargs)
        self._bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self._activation = nn.LeakyReLU(0.2) if discriminator else nn.PReLU()
        self._use_act = use_act

    def forward(self, X):
        X = self._cnn(X)
        X = self._bn(X)
        X = self._activation(X) if self._use_act else X
        return X


class UpSample(nn.Module):
    def __init__(self, in_channels, spatial_scale=1, time_scale=2, **kwargs):
        super().__init__()
        length_scale = width_scale = spatial_scale
        self._cnn = nn.Conv3d(in_channels, in_channels, **kwargs)
        self._upsample = nn.Upsample(
            scale_factor=(length_scale, width_scale, time_scale), mode="nearest"
        )
        self._prelu = nn.PReLU(num_parameters=in_channels)

    def forward(self, X):
        X = self._cnn(X)
        X = self._upsample(X)
        X = self._prelu(X)
        return X


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self._blck_1 = ConvBlock(in_channels, in_channels, **kwargs)
        self._blck_2 = ConvBlock(in_channels, in_channels, use_act=False, **kwargs)

    def forward(self, X):
        y = self._blck_1(X)
        y = self._blck_2(y)
        return y + X


def _create_residual(out_channels, num_blocks, **kwargs):
    residuals = [ResidualBlock(out_channels, **kwargs) for _ in range(num_blocks)]
    return residuals


class Generator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=64,
        num_blocks=16,
        spatial_scale=1,
        time_scale=2,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super().__init__()
        self._initial = ConvBlock(
            in_channels, out_channels, use_bn=False, kernel_size=9, stride=1, padding=4
        )
        self._residual_block = nn.Sequential(
            *_create_residual(
                out_channels,
                num_blocks,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        self._conv_block = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._upsample = nn.Sequential(
            UpSample(
                out_channels,
                spatial_scale,
                time_scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            UpSample(
                out_channels,
                spatial_scale,
                time_scale,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )
        self._cnn = nn.Conv3d(
            out_channels, in_channels, kernel_size=9, stride=1, padding=4
        )

    def forward(self, X):
        initial = self._initial(X)
        X = self._residual_block(initial)
        X = self._conv_block(X) + initial
        X = self._upsample(X)
        X = self._cnn(X)
        X = torch.tanh(X)
        return X


def _create_disc_convblocks(in_channels, features):
    blocks = []
    for index, feature in enumerate(features):
        blocks.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=(in_channels := feature),
                discriminator=True,
                use_act=True,
                use_bn=False if index == 0 else True,
                kernel_size=3,
                stride=1 + index % 2,
                padding=1,
            )
        )
    return blocks


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        out_features = features[-1]
        self._conv_blocks = nn.Sequential(
            *_create_disc_convblocks(in_channels, features)
        )
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((6, 6, 6)),
            nn.Flatten(),
            nn.Linear(out_features * 6 * 6 * 6, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1, bias=True),
        )

    def forward(self, X):
        X = self._conv_blocks(X)
        X = self._classifier(X)
        return X


if __name__ == "__main__":
    sample = torch.zeros((4, 1, 8, 8, 8))
    generator = Generator(in_channels=1)
    X = generator(sample)
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {X.shape}")

    discriminator = Discriminator(1)

    y_hat = discriminator(X)
    print(f"Y_HAT shape: {y_hat.shape}")
    print(discriminator)
