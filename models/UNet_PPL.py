import torch.nn as nn


class _ConvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        Block performing Conv followed by BN, ReLU

        The output feature map size remains the same as
        that of input.
        :param in_ch: Number of input channels
        :param kernel_size: Size of filter
        """
        super(_ConvBNReluBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layers(x)
        return x


class _UpconvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        Block performing UpConv followed by BN, ReLU
        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param kernel_size: Size of filter
        """
        super(_UpconvBNReluBlock, self).__init__()
        self.layers = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch,
                                                       kernel_size),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layers(x)
        return x


class _DoubleConvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(_DoubleConvBNReluBlock, self).__init__()
        self.block1 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)
        self.block2 = _ConvBNReluBlock(out_ch, out_ch, kernel_size)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        return out1, out2


class _TripleConvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(_TripleConvBNReluBlock, self).__init__()
        self.block1 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)
        self.block2 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)
        self.block3 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        return out2, out3


class UNetPPL(nn.Module):
    def __init__(self, n_classes):
        super(UNetPPL, self).__init__()
        self.input = nn.Sequential(nn.Conv2d(3, 64, 3),
                                   nn.ReLU())
        self.enc1 = nn.Sequential(_ConvBNReluBlock(64, 64, 3),
                                  nn.MaxPool2d(2, 2))
        self.enc2 = _TripleConvBNReluBlock(64, 128, 3)
        self.enc3 = _TripleConvBNReluBlock(128, 256, 3)
        self.enc4 = _TripleConvBNReluBlock(256, 512, 3)
        self.enc5 = _TripleConvBNReluBlock(512, 1024, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.dec5 = _DoubleConvBNReluBlock(2048, 1024, 3)
        self.dec4 = _DoubleConvBNReluBlock(1536, 768, 3)
        self.dec3 = _DoubleConvBNReluBlock(1024, 512, 3)
        self.dec2 = _DoubleConvBNReluBlock(768, 256, 3)
        self.dec1 = _DoubleConvBNReluBlock(320, n_classes, 3)

    def forward(self, x):
        inp = self.input(x)

        skip1, out_enc1 = self.enc1(inp)
        out_enc1 = self.pool(out_enc1)

        skip2, out_enc2 = self.enc2(out_enc1)
        out_enc2 = self.pool(out_enc2)

        skip3, out_enc3 = self.enc3(out_enc2)
        out_enc3 = self.pool(out_enc3)

        skip4, out_enc4 = self.enc4(out_enc3)
        out_enc4 = self.pool(out_enc4)

        skip5, out_enc5 = self.enc5(out_enc4)
        out_enc5 = self.pool(out_enc5)
