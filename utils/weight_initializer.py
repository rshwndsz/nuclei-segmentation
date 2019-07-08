# From https://stackoverflow.com/a/49433937/8428730
import torch.nn as nn


def init_weights(m):
    """
    Initialize weights for UNet

    From the paper: Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    arXiv: https://arxiv.org/abs/1502.01852

    Use as:
    ```python
    unet = UNet(3)
    unet.apply(init_weights)
    ```
    :param m: Layer
    """
    if m == nn.Conv2d or m == nn.ConvTranspose2d:
        # See: https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
