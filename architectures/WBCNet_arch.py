import torch
from torch import nn
import torch.nn.functional as F
import functools
from .block import upconv_block, Upsample


####################
# White-box Cartoonization Generators
####################

class ResBlock(nn.Module):
    def __init__(self, in_nf, out_nf=32, slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_nf, out_nf, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_nf, out_nf, 3, 1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)  # True

    def forward(self, inputs):
        x = self.conv2(self.leaky_relu(self.conv1(inputs)))
        return x + inputs


class UnetGeneratorWBC(nn.Module):
    """ UNet Generator as used in Learning to Cartoonize Using White-box
    Cartoon Representations for image to image translation
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791-supp.pdf
    """
    def __init__(self, nf=32, mode='pt', slope=0.2):
        super(UnetGeneratorWBC, self).__init__()

        self.mode = mode

        self.conv = nn.Conv2d(3, nf, 7, 1, padding=3)  # k7n32s1, 256,256
        if mode == 'tf':
            self.conv_1 = nn.Conv2d(nf, nf, 3, stride=2, padding=0)  # k3n32s2, 128,128
        else:
            self.conv_1 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)  # k3n32s2, 128,128
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, padding=1)  # k3n64s1, 128,128

        if mode == 'tf':
            self.conv_3 = nn.Conv2d(nf*2, nf*2, 3, stride=2, padding=0)  # k3n64s2, 64,64
        else:
            self.conv_3 = nn.Conv2d(nf*2, nf*2, 3, stride=2, padding=1)  # k3n64s2, 64,64

        self.conv_4 = nn.Conv2d(nf*2, nf*4, 3, 1, padding=1)  # k3n128s1, 64,64

        # K3n128s1, 4 residual blocks
        self.block_0 = ResBlock(nf*4, nf*4, slope=slope)
        self.block_1 = ResBlock(nf*4, nf*4, slope=slope)
        self.block_2 = ResBlock(nf*4, nf*4, slope=slope)
        self.block_3 = ResBlock(nf*4, nf*4, slope=slope)

        self.conv_5 = nn.Conv2d(nf*4, nf*2, 3, 1, padding=1)  # k3n128s1, 64,64
        self.conv_6 = nn.Conv2d(nf*2, nf*2, 3, 1, padding=1)  # k3n64s1, 64,64
        self.conv_7 = nn.Conv2d(nf*2, nf, 3, 1, padding=1)  # k3n64s1, 64,64
        self.conv_8 = nn.Conv2d(nf, nf, 3, 1, padding=1)  # k3n32s1, 64,64
        self.conv_9 = nn.Conv2d(nf, 3, 7, 1, padding=3)  # k7n3s1, 64,64

        # activations
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope, inplace=False)  # True

        # bilinear resize
        if mode == 'tf':
            self.upsample = Upsample_2xBil_TF()
        else:
            self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # self.act = nn.Tanh() # final output activation

    def forward(self, x):
        # initial feature extraction
        x0 = self.conv(x)
        x0 = self.leaky_relu(x0)  # 256, 256, 32

        # conv block 1
        if self.mode == 'tf':
            x1 = self.conv_1(tf_same_padding(x0))
        else:
            x1 = self.conv_1(x0)
        x1 = self.leaky_relu(x1)
        x1 = self.conv_2(x1)
        x1 = self.leaky_relu(x1)  # 128, 128, 64

        # conv block 2
        if self.mode == 'tf':
            x2 = self.conv_3(tf_same_padding(x1))
        else:
            x2 = self.conv_3(x1)
        x2 = self.leaky_relu(x2)
        x2 = self.conv_4(x2)
        x2 = self.leaky_relu(x2)  # 64, 64, 128

        # residual block
        x2 = self.block_3(self.block_2(self.block_1(self.block_0(x2)))) # 64, 64, 128

        x2 = self.conv_5(x2)
        x2 = self.leaky_relu(x2)  # 64, 64, 64

        # upconv block 1
        x3 = self.upsample(x2)
        x3 = self.conv_6(x3 + x1)
        x3 = self.leaky_relu(x3)
        x3 = self.conv_7(x3)
        x3 = self.leaky_relu(x3)  # 128, 128, 32

        # upconv block 2
        x4 = self.upsample(x3)
        x4 = self.conv_8(x4 + x0)
        x4 = self.leaky_relu(x4)
        x4 = self.conv_9(x4)  # 256, 256, 32

        # x4 = torch.clamp(x4, -1, 1)
        # return self.act(x4)
        return x4


class Upsample_2xBil_TF(nn.Module):
    def __init__(self):
        super(Upsample_2xBil_TF, self).__init__()

    def forward(self, x):
        return tf_2xupsample_bilinear(x)


def tf_2xupsample_bilinear(x):
    b, c, h, w = x.shape
    out = torch.zeros(b, c, h*2, w*2).to(x.device)
    out[:, :, ::2, ::2] = x
    padded = F.pad(x, (0, 1, 0, 1), mode='replicate')
    out[:, :, 1::2, ::2] = (
        padded[:, :, :-1, :-1] + padded[:, :, 1:, :-1])/2
    out[:, :, ::2, 1::2] = (
        padded[:, :, :-1, :-1] + padded[:, :, :-1, 1:])/2
    out[:, :, 1::2, 1::2] = (
        padded[:, :, :-1, :-1] + padded[:, :, 1:, 1:])/2
    return out


def tf_same_padding(x, k_size=3):
    j = k_size//2
    return F.pad(x, (j-1, j, j-1, j))