import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_pool=True,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super().__init__()
        self._conv_block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.Identity(),
        )

    def forward(self, X):
        X = self._conv_block(X)
        return X


def _create_conv_layers(in_channels, features):
    layers = []
    for index, feature in enumerate(features):
        layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=feature,
                max_pool=True if (((index + 1) % 3 == 1) and index > 0) else False,
            )
        )
        in_channels = feature
    return layers


class Vgg3D(nn.Module):
    def __init__(self, in_channels, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        out_features = features[-1]
        self._feature_extractor = nn.Sequential(
            *_create_conv_layers(in_channels, features)
        )
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((6, 6, 6)),
            nn.Flatten(),
            nn.Linear(out_features * 6 * 6 * 6, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1, bias=True),
        )

    def forward(self, X):
        X = self._feature_extractor(X)
        X = self._classifier(X)
        return X

    @property
    def classifier(self):
        return self._classifier

    @property
    def feature_extractor(self):
        return self._feature_extractor


if __name__ == "__main__":
    channels = 1
    size = 8
    batch_size = 1
    X = torch.zeros((batch_size, channels, size, size, size))
    vgg3d = Vgg3D(in_channels=channels)
    out = vgg3d(X)
    print(out.shape)
