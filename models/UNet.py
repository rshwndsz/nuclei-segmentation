from torch import nn
import torch
from collections import OrderedDict


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(_EncoderBlock, self).__init__()
        layers = OrderedDict([
            ('CONV1', nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding_mode='valid')),
            ('BN1', nn.BatchNorm2d(out_channels)),
            ('RELU1', nn.ReLU(inplace=True)),
            ('CONV2', nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding_mode='valid')),
            ('BN1', nn.BatchNorm2d(out_channels)),
            ('RELU2', nn.ReLU(inplace=True))
        ])
        if dropout:
            layers.update({'DROPOUT': nn.Dropout(p=0.2)})
        layers.update({'MAX_POOL': nn.MaxPool2d(kernel_size=2, stride=2)})
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encode(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        layers = OrderedDict([
            ('CONV1', nn.Conv2d(in_channels, mid_channels,
                                kernel_size=3, padding_mode='valid')),
            ('BN1', nn.BatchNorm2d(mid_channels)),
            ('RELU1', nn.ReLU(inplace=True)),
            ('CONV2', nn.Conv2d(mid_channels, mid_channels,
                                kernel_size=3, padding_mode='valid')),
            ('BN1', nn.BatchNorm2d(mid_channels)),
            ('RELU2', nn.ReLU(inplace=True))
        ])
        layers.update({'TR_CONV': nn.ConvTranspose2d(mid_channels, out_channels,
                                                     kernel_size=2, stride=2)})
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decode(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.input = _EncoderBlock(3, 64)
        self.enc1 = _EncoderBlock(64, 128)
        self.enc2 = _EncoderBlock(128, 256)
        self.enc3 = _EncoderBlock(256, 512)

        self.center = _DecoderBlock(512, 1024, 512)

        self.dec3 = _DecoderBlock(1024, 512, 256)
        self.dec2 = _DecoderBlock(512, 256, 128)
        self.dec1 = _DecoderBlock(256, 128, 64)
        self.final = nn.Sequential(OrderedDict([
            ('CONV1', nn.Conv2d(128, 64,
                                kernel_size=3, padding_mode='valid')),
            ('BN1', nn.BatchNorm2d(64)),
            ('RELU1', nn.ReLU(inplace=True)),
            ('CONV2', nn.Conv2d(64, 64,
                                kernel_size=3, padding_mode='valid')),
            ('BN1', nn.BatchNorm2d(64)),
            ('RELU2', nn.ReLU(inplace=True)),
        ]))
        self.output = nn.Conv2d(64, n_classes,
                                kernel_size=3, padding_mode='valid')

    @staticmethod
    def crop(layer, target_size):
        # See: https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py#L113
        layer_height, layer_width = layer.size()[2:]
        diff_x = (layer_height - target_size[0]) // 2
        diff_y = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y:(diff_y+target_size[0]), diff_x:(diff_x+target_size[1])
        ]

    def forward(self, x):
        input = self.input(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(x)
        enc3 = self.enc3(x)

        center = self.center(x)
        return x
