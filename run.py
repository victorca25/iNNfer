import argparse
import os.path as osp
import torch

from utils.utils import (mod2normal, get_models_paths, get_images_paths,
                    read_img, np2tensor, tensor2np, color_fix, save_img,
                    save_img_comp, extract_patches_2d, recompose_tensor,
                    linear_resize, swa2normal, guided_filter, modcrop)
from utils.defaults import get_network_G_config
from architectures import get_network


class nullcast():
    #nullcontext:
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *excinfo):
        pass


class Model:
    def __init__(self, model_path, arch=None, scale=None,
            in_nc=3, out_nc=3, device='cpu', meval=True,
            strict=True, chop=True):
        self.model_path = model_path
        self.arch = arch
        self.scale = scale
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.device = device
        self.model = None
        self.eval = meval
        self.strict = strict
        self.chop = chop
        self.load_model()

    def load_model(self):
        if self.arch == 'ts':
            self.model = torch.jit.load(
                osp.join(self.model_path)).eval().to(self.device)
        else:
            state_dict = torch.load(self.model_path)

            # convert from SWA to regular model if needed
            if 'n_averaged' in state_dict:
                state_dict = swa2normal(state_dict)

            if self.arch == 'infer':
                if 'SCPA_trunk.0.conv1_a.weight' in state_dict:
                    # pan model
                    self.arch = 'pan'
                elif 'model.1.sub.0.res.0.weight' in state_dict:
                    # srgan
                    self.arch = 'srgan'
                elif 'conv_first.weight' in state_dict:
                    # self.arch = 'mesrgan'
                    # convert msergan to esrgan
                    state_dict = mod2normal(state_dict)
                    self.arch = 'esrgan'
                elif 'model.0.weight' in state_dict:
                    # regular esrgan
                    self.arch = 'esrgan'
                elif 'CFEM.0.weight' in state_dict:
                    # ppon model
                    self.arch = 'ppon'
                elif 'conv_9.weight' in state_dict:
                    # wbc UNET (TODO: validate)
                    self.arch = 'wbcunet'
                else:
                    raise Exception("Could not infer model parameters.")
                net_params = self.infer_params(state_dict)
            else:
                # use defaults
                net_dict = {}
                if not self.scale:
                    self.scale = 1
                if 'wbcunet' in self.arch and "_tf" in self.arch:
                    self.arch = self.arch.replace("_tf", "")
                    net_dict["mode"] = "tf"
                elif 'wbcunet' in self.arch:
                    net_dict["mode"] = "pt"

                net_dict['type'] = self.arch
                net_params = get_network_G_config(
                    net_dict, self.scale)

            # define network
            net = get_network(net_params)

            # load state dict, set to eval and stop grad
            net.load_state_dict(state_dict, strict=self.strict)
            del state_dict
            for k, v in net.named_parameters():
                v.requires_grad = False

            if self.eval:
                net.eval()

            self.model = net.to(self.device)

    def infer_params(self, state_dict):
        # extract model information
        if self.arch in ('esrgan', 'srgan'):
            scale2x = 0
            scalemin = 6
            n_uplayer = 0
            if self.arch == 'esrgan':
                plus = False

            #TODO
            # print(list(state_dict))

            for block in list(state_dict):
                parts = block.split(".")
                n_parts = len(parts)
                if n_parts == 5 and parts[2] == "sub":
                    # num. rrdb (or res) blocks from last conv. layer before upscales
                    nb = int(parts[3])
                elif n_parts == 3:
                    # upscale blocks
                    part_num = int(parts[1])
                    if (part_num > scalemin
                        and parts[0] == "model"
                        and parts[2] == "weight"):
                        # num. 2x upsample blocks
                        scale2x += 1
                    if part_num > n_uplayer:
                        # fetch out_nc from last layer shape
                        n_uplayer = part_num
                        out_nc = state_dict[block].shape[0]
                if self.arch == 'esrgan':
                    if not plus and "conv1x1" in block:
                        plus = True
            nf = state_dict["model.0.weight"].shape[0]
            self.in_nc = state_dict["model.0.weight"].shape[1]
            self.out_nc = out_nc
            self.scale = 2 ** scale2x

            net_dict = {
                'type': self.arch,
                'in_nc': self.in_nc,
                'out_nc': self.out_nc,
                'nf': nf,
                'nb': nb,
            }
            if self.arch == 'esrgan':
                net_dict['plus'] = plus
        elif self.arch == 'wbcunet':
            self.scale = 1
            net_dict = {
                'type': self.arch,
                'mode': 'pt',  # 'tf'  # TODO
                'nf': state_dict["conv.weight"].shape[0],
            }
        elif self.arch in ['ppon', 'pan']:
            # custom params inference TBD
            net_dict = {
                'type': self.arch,
                'in_nc': self.in_nc,
                'out_nc': self.out_nc,
            }

        return get_network_G_config(net_dict, self.scale)

    def chop_forward(self, data, patch_size=200, step=1.0):
        """ Chop forward function used in test time.
        Converts large images into patches of size (patch_size, patch_size).
        Make sure the patch size is small enough that your GPU memory is sufficient.
        Examples: patch_size = 200 for BlindSR, 64 for ABPN
        """
        batch_size, channels, img_height, img_width = data.size()
        # if (patch_size * (1.0 - step)) % 1 < 0.5:
        #     patch_size += 1
        patch_size = min(img_height, img_width, patch_size)

        img_patches = extract_patches_2d(img=data, 
                                        patch_shape=(patch_size, patch_size), 
                                        step=[step, step], 
                                        batch_first=True).squeeze(0)
        
        n_patches = img_patches.size(0)
        highres_patches = []

        with self.get_torch_ctx():
            for p in range(n_patches):
                # print(p)
                lowres_input = img_patches[p:p + 1]
                prediction = self.model(lowres_input)
                if self.arch == 'ppon':
                    prediction = prediction[2]
                if self.arch == 'ts':
                    # fix for CUDA out of memory.
                    prediction = prediction.detach().cpu()
                highres_patches.append(prediction)
                torch.cuda.empty_cache()

        highres_patches = torch.cat(highres_patches, 0)

        return recompose_tensor(
            highres_patches, img_height, img_width, step=step, scale=self.scale)

    def get_torch_ctx(self):
        if self.arch == 'ts' and float(torch.__version__[:3]) < 1.8:
            # issue with torchscript: RuntimeError: CUDA driver error: a PTX JIT compilation failed
            # https://github.com/pytorch/pytorch/issues/47304
            return nullcast()
        return torch.no_grad()

    def __call__(self, data):
        if self.chop:
            t_out = self.chop_forward(
                patch_size=200,  # 100
                step=0.5,  # 0.9
                data=data,)
        else:
            with self.get_torch_ctx():
                t_out = self.model(data)
            if self.arch == 'ppon':
                t_out = t_out[2]

        torch.cuda.empty_cache()

        return t_out



def parse_models(models_paths, scales_list=None):

    model_chain = models_paths.split("+") if "+" in models_paths else models_paths.split(">")

    all_models = get_models_paths("./models")

    full_chain = []
    for model_path in model_chain:
        full_chain.append(check_model_path(model_path, all_models))

    # try to get model scale from model name
    if not scales_list:
        scales_list = [None] * len(full_chain)
        rlt_scales = []
        for m, sc in zip(full_chain, scales_list):
            rlt_scales.append(
                get_scale_name(m, sc))
        scales_list = rlt_scales
    else:
        if len(scales_list) != len(model_chain):
            raise ValueError(
                f"The num. of scales {len(scales_list)} is != from number of models {len(model_chain)}")

    return full_chain, scales_list


def check_model_path(model_path, all_models=None):
    # check if model exists in absolute path or ./models
    if not osp.isfile(model_path):
        model_path_a = osp.join("models", model_path)
        if not osp.isfile(model_path_a):
            # partial name search in ./models
            if all_models:
                m_list = []
                for m in all_models:
                    # if str(m).lower().find(str(model_path.lower())) >= 0:
                    if str(model_path.lower()) in str(m).lower():
                        m_list.append(m)
                if len(m_list) > 1:
                    raise ValueError(
                        f"Filter {model_path} returned multiple models: {m_list}.")
                model_path = m_list[0]
            else:
                raise ValueError(f"Model {model_path} not found.")
        else:
            model_path = model_path_a
    return model_path


def get_scale_name(model_path, scale=None):
    """ try to get model scale from model name"""

    rlt_scale = None
    scale_name = str(osp.basename(model_path)[0:2]).lower()
    if 'x' in scale_name:
        try:
            rlt_scale = int(scale_name.replace('x', ''))
        except ValueError:
            rlt_scale = None

    if scale:
        if rlt_scale and (scale != rlt_scale):
            print(f"Warning: possible model scale mismatch on {model_path}")
        return scale
    return rlt_scale





pix2pix_extras = {
    'meval': False,  # pix2pix could produce slightly better results with eval=False (uses norm layers params)
    'strict': True,
    'normalize': True,  # pix2pix and cyclegan use normalized images
    }

cyglegan_extras = {
    'meval': True,
    'strict': False,  # to ignore batch statistics that were enabled models trained with Pytorch < 0.4.0
    'normalize': True,  # pix2pix and cyclegan use normalized images
    }

default_extras = {
    'meval': True,
    'strict': True,
    'normalize': False,
    }





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-models', '-m', type=str, required=True, help='Path to models.')
    parser.add_argument('-arch', '-a', type=str, required=False, default='infer', help='Model architecture.')
    parser.add_argument('-input', '-i', type=str, required=False, default='./input', help='Path to read input images.')
    parser.add_argument('-output', '-o', type=str, required=False, default='./output', help='Path to save output images.')
    parser.add_argument('-scale', '-s', type=str, required=False, default='-1', help='Model scaling factor.')
    parser.add_argument('-cf', required=False, action='store_true', help='Use color correction if enabled.')
    parser.add_argument('-comp', required=False, action='store_true', help='Save as comparison images if enabled.')
    parser.add_argument('-no_gpu', '-cpu', required=False, action='store_false', help='Run in CPU if enabled.')
    parser.add_argument('-no_fp16', required=False, action='store_false', help='Disable fp16 mode if needed.')
    parser.add_argument('-norm', required=False, action='store_true', help='Normalizes images in range [-1,1] if set, else [0,1].')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    gpu = args.no_gpu  # TODO: fp16 error with cpu: RuntimeError: "unfolded2d_copy" not implemented for 'Half'
    # TODO: all these options should be configurable
    if args.arch == 'ts':
        # TODO: not working with torchscript unless model was traced with fp16
        fp16 = False # True
    else:
        fp16 = args.no_fp16 and gpu

    use_guided_filter = False
    use_modcrop = False
    if 'unet_' in args.arch or 'p2p_' in args.arch:
        defaults = pix2pix_extras
        chop = False  # tmp, could chop to unet size
        if '512' in args.arch:
            resize = 512
        elif '256' in args.arch:
            resize = 256
        elif '128' in args.arch:
            resize = 128
    elif 'resnet_' in args.arch or 'cg_' in args.arch:
        defaults = cyglegan_extras
        chop = True
        resize = False
    elif 'wbc' in args.arch or 'wbc' in args.models:
        if 'tf' in args.arch or 'tf' in args.models:
            args.arch = "wbcunet_tf"
        else:
            args.arch = "wbcunet"
        defaults = pix2pix_extras
        chop = False  # True
        resize = False
        use_guided_filter = True
        use_modcrop = True
    else:
        defaults = default_extras
        resize = False
        chop = True

    meval = defaults['meval']
    strict = defaults['strict']
    normalize = defaults['normalize'] or args.norm


    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor if gpu else torch.HalfTensor)
    device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu') 

    cf = args.cf  # color fix
    comp = args.comp  # save comparison images
    model_path = args.models
    output_dir = args.output
    # TODO: chain scales
    scale = args.scale if args.scale != -1 else None
    # TODO: chain archs

    model_chain, scale_chain = parse_models(model_path)

    models = []
    for mc, sc in zip(model_chain, scale_chain):
        models.append(
            Model(
                mc, args.arch, sc, device=device, meval=meval, strict=strict, chop=chop))

    images = get_images_paths(args.input)

    for image_path in images:

        img_name = osp.splitext(osp.basename(image_path))[0]
        img = read_img(image_path)

        # if not isinstance(img, np.ndarray):
        if img is None:
            print(f'Error reading image {image_path}, skipping.')
            continue
        
        # TODO: can pad|resize|crop images to next size accepted by network
        if resize:
            img = linear_resize(img, resize)

        if use_modcrop:
            img = modcrop(img, 4)

        t_img = np2tensor(img, normalize=normalize).to(device)
        t_img = t_img.half() if fp16 else t_img

        t_out = t_img.clone()
        for mod in models:
            t_out = mod(t_out)
            if use_guided_filter:
                # note: r can be configured here to control details in results
                t_out = guided_filter(t_img, t_out, r=1, eps=5e-3)

        img_out = tensor2np(t_out.detach(), denormalize=normalize)

        if cf:
            img_out = color_fix(img, img_out)

        # save images
        save_img_path = osp.join(
            output_dir, f'{img_name:s}.png')
        if comp:
            save_img_comp([img, img_out], save_img_path)
        else:
            save_img(img_out, save_img_path)



if __name__ == '__main__':
    main()
