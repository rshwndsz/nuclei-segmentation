import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        """
        Encoder block of the UNet structure

        Contains 2 Conv-BN-ReLU blocks followed by a max pooling layer
        Can have dropout for regularization if dropout=True
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param dropout: If True, dropout is included
        """
        super(_EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=0.2) if dropout else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.encode(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        """
        Decoder block of the UNet Structure

        Contains 2 Conv-BN-ReLU blocks followed by a Transpose convolution
        :param in_channels: Number of input channels
        :param mid_channels: Number of intermediate channels
        :param out_channels: Number of output channels
        """
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.decode(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes):
        """
        Base UNet

        Paper: https://arxiv.org/abs/1505.04597
        :param n_classes: Number of output classes
        """
        super(UNet, self).__init__()
        self.input = _EncoderBlock(3, 64)
        self.enc1 = _EncoderBlock(64, 128)
        self.enc2 = _EncoderBlock(128, 256)
        self.enc3 = _EncoderBlock(256, 512, dropout=True)

        self.center = _DecoderBlock(512, 1024, 512)

        self.dec3 = _DecoderBlock(1024, 512, 256)
        self.dec2 = _DecoderBlock(512, 256, 128)
        self.dec1 = _DecoderBlock(256, 128, 64)
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        inp = self.input(x)
        enc1 = self.enc1(inp)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        center = self.center(enc3)

        crop1 = F.interpolate(enc3, center.size()[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([center, crop1], 1))

        crop2 = F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([dec3, crop2], 1))

        crop3 = F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([dec2, crop3], 1))

        crop4 = F.interpolate(inp, dec1.size()[2:], mode='bilinear', align_corners=False)
        fin = self.final(torch.cat([dec1, crop4], 1))

        out = self.output(fin)
        out_upsampled = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=False)
        return out_upsampled
