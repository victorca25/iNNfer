



def get_network(opt_net):
    """Instantiate the network with configuration"""

    kind = opt_net.pop('type').lower()

    # generators
    if kind == 'sr_resnet':
        from . import SRResNet_arch
        net = SRResNet_arch.SRResNet
    elif kind == 'rrdb_net':  # ESRGAN
        from . import RRDBNet_arch
        net = RRDBNet_arch.RRDBNet
    elif kind == 'mrrdb_net':  # Modified ESRGAN
        from . import RRDBNet_arch
        net = RRDBNet_arch.MRRDBNet
    elif kind == 'ppon':
        from . import PPON_arch
        net = PPON_arch.PPON
    elif kind == 'pan_net':
        from . import PAN_arch
        net = PAN_arch.PAN
    elif kind == 'unet_net':
        from . import UNet_arch
        net = UNet_arch.UnetGenerator
    elif kind == 'resnet_net':
        from . import ResNet_arch
        net = ResNet_arch.ResnetGenerator
    elif kind == 'wbcunet_net':
        from . import WBCNet_arch
        net = WBCNet_arch.UnetGeneratorWBC
    else:
        raise NotImplementedError('Model [{:s}] not recognized'.format(kind))

    net = net(**opt_net)

    return net
