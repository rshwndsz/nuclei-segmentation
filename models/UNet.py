import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class _DoubleConvBlock(nn.Module):
    # TODO: Use this as it is more modular
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU(inplace=True)):
        """
        2 unpadded convolutions each followed by a ReLU.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel
        :param activation: activation function (ReLU by default)
        """
        super(_DoubleConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_p=0):
        super(_EncoderBlock, self).__init__()
        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(p=drop_p) if drop_p else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        logger.info('_EncoderBlock: START')
        x = self.encode(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        logger.info('_EncoderBlock: FINISH')
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        logger.info('_DecoderBlock: START')
        x = self.decode(x)
        logger.info('_DecoderBlock: FINISH')
        return x


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # TODO: Try with dropout
        self.input = _EncoderBlock(3, 64)
        self.enc1 = _EncoderBlock(64, 128)
        self.enc2 = _EncoderBlock(128, 256)
        self.enc3 = _EncoderBlock(256, 512)

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

    @staticmethod
    def center_crop(layer, target_size):
        # TODO: find cleaner implementation
        # See: https://discuss.pytorch.org/t/unet-implementation/426
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2       # number of missing pixels in x-dim
        diff_x = (layer_width - target_size[1]) // 2        # number of missing pixels in y-dim
        return layer[
            :, :, diff_y:(diff_y+target_size[0]), diff_x:(diff_x+target_size[1])
        ]

    def forward(self, x):
        inp = self.input(x)
        enc1 = self.enc1(inp)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        logger.info('Before center')
        center = self.center(enc3)
        logger.info('After center')

        crop1 = F.interpolate(enc3, center.size()[2:], mode='bilinear', align_corners=False)
        logger.info('After crop1')
        logger.info('center: {}'.format(center.size()))
        logger.info('crop1: {}'.format(crop1.size()))
        dec3 = self.dec3(torch.cat([center, crop1], 1))
        logger.info('After dec3')
        crop2 = F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([dec3, crop2], 1))
        crop3 = F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([dec2, crop3], 1))

        logger.info('before final')
        logger.info('dec1: {}'.format(dec1.size()))
        crop4 = F.interpolate(inp, dec1.size()[2:], mode='bilinear', align_corners=False)
        fin = self.final(torch.cat([dec1, crop4], 1))
        logger.info('after final')
        logger.info('fin: {}'.format(fin.size()))
        out = self.output(fin)
        logger.info(f'after out, {out.shape}')
        print('out: {}'.format(out))
        out_upsampled = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=False)
        return out
