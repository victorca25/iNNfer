

def get_network_G_config(network_G, scale):
    kind_G = None
    scale = int(scale)
    if isinstance(network_G, str):
        kind_G = network_G.lower()
        network_G = {}
    elif isinstance(network_G, dict):
        if 'which_model_G' in network_G:
            which_model = 'which_model_G'
        elif 'type' in network_G:
            which_model = 'type'
        kind_G = network_G.pop(which_model).lower()

    full_network_G = {}
    # full_network_G['strict'] = network_G.pop('strict', False) # True | False: whether to load the model in strict mode or not

    # SR networks
    if kind_G in ('rrdb_net', 'esrgan', 'evsrgan', 'esrgan-lite'):
        # ESRGAN (or EVSRGAN):
        full_network_G['type'] = "rrdb_net" # RRDB_net (original ESRGAN arch)
        full_network_G['norm_type'] = network_G.pop('norm_type', None)  # "instance" normalization, "batch" normalization or no norm
        full_network_G['mode'] = network_G.pop('mode', "CNA")  # CNA: conv->norm->act, NAC: norm->act->conv
        if kind_G == 'esrgan-lite':
            full_network_G['nf'] = network_G.pop('nf', 32)  # number of filters in the first conv layer
            full_network_G['nb'] = network_G.pop('nb', 12)  # number of RRDB blocks
        else:
            full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
            full_network_G['nb'] = network_G.pop('nb', 23)  # number of RRDB blocks
        full_network_G['nr'] = network_G.pop('nr', 3)  #  number of residual layers in each RRDB block
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['gc'] = network_G.pop('gc', 32)  #
        if kind_G == 'evsrgan':
            full_network_G['convtype'] = network_G.pop('convtype', "Conv3D")  # Conv3D for video
        else:
            full_network_G['convtype'] = network_G.pop('convtype', "Conv2D")  # Conv2D | PartialConv2D | DeformConv2D | Conv3D
        full_network_G['act_type'] = network_G.pop('net_act', None) or network_G.pop('act_type', "leakyrelu")  # swish | leakyrelu
        full_network_G['gaussian_noise'] = network_G.pop('gaussian', True)  # add gaussian noise in the net latent # True | False
        full_network_G['plus'] = network_G.pop('plus', False)  # use the ESRGAN+ modifications # true | false
        full_network_G['finalact'] = network_G.pop('finalact', None)  # Activation function, ie use "tanh" to make outputs fit in [-1, 1] range. Default = None. Coordinate with znorm.
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "upconv") # the type of upsample to use
    elif kind_G in ('mrrdb_net', 'mesrgan'):
        # ESRGAN modified arch:
        full_network_G['type'] = "mrrdb_net" # MRRDB_net (modified/"new" arch) | sr_resnet
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 24)  # number of RRDB blocks
        full_network_G['gc'] = network_G.pop('gc', 32)  #
    elif kind_G in ('sr_resnet', 'srresnet', 'srgan'):
        # SRGAN:
        full_network_G['type'] = "sr_resnet"  # SRResNet
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 16)  # number of RRDB blocks
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['norm_type'] = network_G.pop('norm_type', None)  # "instance" normalization, "batch" normalization or no norm
        full_network_G['act_type'] = network_G.pop('net_act', None) or network_G.pop('act_type', "relu")  # swish | relu | leakyrelu
        full_network_G['mode'] = network_G.pop('mode', "CNA")  # CNA: conv->norm->act, NAC: norm->act->conv
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "pixelshuffle") # the type of upsample to use
        full_network_G['convtype'] = network_G.pop('convtype', "Conv2D")  # Conv2D | PartialConv2D | DeformConv2D | Conv3D
        full_network_G['finalact'] = network_G.pop('finalact', None)  # Activation function, ie use "tanh" to make outputs fit in [-1, 1] range. Default = None. Coordinate with znorm.
        full_network_G['res_scale'] = network_G.pop('res_scale', 1)
    elif 'ppon' in kind_G:
        # PPON:
        full_network_G['type'] = "ppon"  # RRDB_net (original ESRGAN arch)
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 64)  # number of filters in the first conv layer
        full_network_G['nb'] = network_G.pop('nb', 24)  # number of RRDB blocks
        full_network_G['upscale'] = network_G.pop('scale', scale)
        full_network_G['act_type'] = network_G.pop('net_act', None) or network_G.pop('act_type', "leakyrelu")  # swish | leakyrelu
        full_network_G['alpha'] = network_G.pop('alpha', 1)  # interpolation value between percepual and structure
    elif kind_G in ('pan_net', 'pan'):
        # PAN:
        full_network_G['type'] = "pan_net"  # PAN_net
        full_network_G['in_nc'] = network_G.pop('in_nc', 3) # num. of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['out_nc'] = network_G.pop('out_nc', 3) # num. of output image channels: 3 for RGB and 1 for grayscale
        full_network_G['nf'] = network_G.pop('nf', 40)  # number of filters in each conv layer
        full_network_G['unf'] = network_G.pop('unf', 24)  # number of filters during upscale
        full_network_G['nb'] = network_G.pop('nb', 16)  # number of blocks
        full_network_G['scale'] = network_G.pop('scale', scale)
        full_network_G['self_attention'] = network_G.pop('self_attention', True)
        full_network_G['double_scpa'] = network_G.pop('double_scpa', False)
        full_network_G['ups_inter_mode'] = network_G.pop('ups_inter_mode', "nearest")
    # image to image translation
    elif 'wbcunet' in kind_G:
        # WBC
        full_network_G['type'] = "wbcunet_net"
        full_network_G['nf'] = network_G.pop('nf', 32)
        if 'tf' in kind_G:
            full_network_G['mode'] = 'tf'
        else:
            full_network_G['mode'] = network_G.pop('mode', 'pt')
    elif 'unet' in kind_G or 'p2p' in kind_G:
        # UNET:
        full_network_G['type'] = "unet_net"
        full_network_G['input_nc'] = network_G.pop('in_nc', 3) # # of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['output_nc'] = network_G.pop('out_nc', 3) # # of output image channels: 3 for RGB and 1 for grayscale
        if kind_G in ('unet_128', 'p2p_128'):
            full_network_G['num_downs'] = network_G.pop('num_downs', 7) # for 'unet_128' (for 128x128 input images)
        elif kind_G in ('unet_256', 'p2p_256'):
            full_network_G['num_downs'] = network_G.pop('num_downs', 8) # for 'unet_256' (for 256x256 input images)
        else:
            full_network_G['num_downs'] = network_G.pop('num_downs', 8) #7 for 'unet_128' (for 128x128 input images) | 8 for 'unet_256' (for 256x256 input images)
        # # check valid crop size for UNET
        # if full_network_G['num_downs'] == 7:
        #     assert crop_size == 128, f'Invalid crop size {crop_size} for UNET config, must be 128'
        # elif full_network_G['num_downs'] == 8:
        #     assert crop_size == 256, f'Invalid crop size {crop_size} for UNET config, must be 256'
        # elif full_network_G['num_downs'] == 9:
        #     assert crop_size == 512, f'Invalid crop size {crop_size} for UNET config, must be 512'
        full_network_G['ngf'] = network_G.pop('ngf', 64) # # of gen filters in the last conv layer
        full_network_G['norm_type'] = network_G.pop('norm_type', "batch") # "instance" normalization or "batch" normalization
        full_network_G['use_dropout'] = network_G.pop('use_dropout', False) # whether to use dropout or not
        #TODO: add:
        # full_network_G['dropout_prob'] = network_G.pop('dropout_prob', 0.5) # the default dropout probability
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "deconv") # deconv | upconv # the type of upsample to use, deconvolution or upsample+convolution
    elif ('resnet' in kind_G and kind_G != 'sr_resnet') or 'cg' in kind_G:
        # RESNET:
        full_network_G['type'] = "resnet_net"
        full_network_G['input_nc'] = network_G.pop('in_nc', 3) # # of input image channels: 3 for RGB and 1 for grayscale
        full_network_G['output_nc'] = network_G.pop('out_nc', 3) # # of output image channels: 3 for RGB and 1 for grayscale
        if kind_G in ('resnet_6blocks', 'resnet_6', 'cg_6'):
            full_network_G['n_blocks'] = network_G.pop('n_blocks', 6)  # 6 for resnet_6blocks (with 6 Resnet blocks) and
        elif kind_G in ('resnet_9blocks', 'resnet_9', 'cg9'):
            full_network_G['n_blocks'] = network_G.pop('n_blocks', 9)  # 9 for resnet_9blocks (with 9 Resnet blocks)
        else:
            full_network_G['n_blocks'] = network_G.pop('n_blocks', 9)  # 6 for resnet_6blocks (with 6 Resnet blocks) and 9 for resnet_9blocks (with 9 Resnet blocks)
        full_network_G['ngf'] = network_G.pop('ngf', 64)  # num. of gen filters in the last conv layer
        full_network_G['norm_type'] = network_G.pop('norm_type', "instance") # "instance" normalization or "batch" normalization
        full_network_G['use_dropout'] = network_G.pop('use_dropout', False) # whether to use dropout or not
        #TODO: add:
        # full_network_G['dropout_prob'] = network_G.pop('dropout_prob', 0.5) # the default dropout probability
        full_network_G['upsample_mode'] = network_G.pop('upsample_mode', "deconv") # deconv | upconv # the type of upsample to use, deconvolution or upsample+convolution
        full_network_G['padding_type'] = network_G.pop('padding_type', "reflect")
    else:
        raise NotImplementedError(f'Generator model [{kind_G:s}] not recognized')

    #TODO: check if any options in network_G went unprocessed
    if bool(network_G):
        print(network_G)

    return full_network_G