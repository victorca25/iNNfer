import torch
import numpy as np


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    # https://github.com/pytorch/pytorch/issues/229
    out: torch.Tensor = image.flip(-3)
    # RGB to BGR #may be faster:
    #out: torch.Tensor = image[[2, 1, 0], :, :]
    return out


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def srgb2linear(srgb, gamma=2.4, th=0.04045):
    """ Convert SRGB images to linear RGB color space.
        To use the formulat, values have to be in the 0 to 1 range,
        for that reason srgb must be in range [0,255], uint8 and:
        signal = input / 255 is applied.
    Parameters:
        gamma (float): gamma correction. The default is 2.4, but 2.2
            approximately matches the power law sensitivity of human vision
        th (float): threshold value for formula. The default is 0.04045,
            which is an approximate, exact value is 0.0404482362771082
    """

    a = 0.055
    att = 12.92
    linear = np.float32(srgb) / 255.0

    return np.where(
        linear<=th, linear/att, np.power((linear+a)/(1+a), gamma))


def linear2srgb(linear, gamma=2.4, th=0.0031308):
    """ Convert linear RGB images to SRGB color space.
    linear must be in range [0,1], float32 """
    a = 0.055
    att = 12.92
    srgb = np.clip(linear.copy(), 0.0, 1.0)

    srgb = np.where(
        srgb<=th, srgb*att, (1+a)*np.power(srgb, 1.0/gamma)-a)

    # return srgb * 255.0
    return np.clip(srgb * 255.0, 0.0, 255).astype(np.uint8)