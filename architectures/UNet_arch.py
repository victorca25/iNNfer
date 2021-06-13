import torch
from torch import nn
import functools
from .block import upconv_block, Upsample


####################
# UNet Generator
####################

class UnetGenerator(nn.Module):
    """Create a Unet-based generator
    U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
    As used by pix2pix: https://arxiv.org/pdf/1611.07004.pdf
    From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_type="batch",
                use_dropout=False, upsample_mode="deconv"):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example:
                                - if |num_downs| == 7 image of size 128x128 will 
                                  become of size 1x1 # at the bottleneck
                                - if |num_downs| == 8, image of size 256x256 will 
                                  become of size 1x1 at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_type       -- normalization layer type
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        if norm_type in ('BN', 'batch'):
            norm_layer = nn.BatchNorm2d
        elif norm_type in ('IN', 'instance'):
            norm_layer = nn.InstanceNorm2d
        else:
            raise NameError("Unknown norm layer")

        # construct unet structure
        
        ## add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, 
            innermost=True, upsample_mode=upsample_mode)
        
        ## add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                norm_layer=norm_layer, use_dropout=use_dropout, upsample_mode=upsample_mode)
        
        ## gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample_mode=upsample_mode)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample_mode=upsample_mode)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample_mode=upsample_mode)
        
        ## add the outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, 
            norm_layer=norm_layer, upsample_mode=upsample_mode)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, upsample_mode="deconv"):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            upsample_mode (str) -- upsampling strategy: deconv (original) | upconv | pixelshuffle
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if type(norm_layer) is functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=inner_nc * 2, out_nc=outer_nc,
                                        kernel_size=3, stride=1, 
                                        act_type=None)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=inner_nc, out_nc=outer_nc,
                                        kernel_size=3, stride=1, 
                                        bias=use_bias, act_type=None)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if upsample_mode=='deconv':
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            elif upsample_mode=='upconv':
                upconv = upconv_block(in_nc=inner_nc * 2, out_nc=outer_nc,
                                        kernel_size=3, stride=1, 
                                        bias=use_bias, act_type=None)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
