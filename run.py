import logging
import sys
from pathlib import Path
from typing import List

import click
import torch
from rich import get_console, print
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TimeRemainingColumn,
)
from rich.traceback import install as install_traceback

from architectures import get_network
from utils.defaults import get_network_G_config
from utils.utils import (
    color_fix,
    extract_patches_2d,
    get_images_paths,
    get_models_paths,
    guided_filter,
    linear_resize,
    mod2normal,
    modcrop,
    np2tensor,
    read_img,
    recompose_tensor,
    save_img,
    save_img_comp,
    swa2normal,
    tensor2np,
)

install_traceback()


class nullcast:
    # nullcontext:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


class Model:
    def __init__(
        self,
        model_path,
        arch=None,
        scale=None,
        in_nc=3,
        out_nc=3,
        device="cpu",
        meval=True,
        strict=True,
        chop=True,
    ):
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
        with get_console().status(f'Loading model [green]"{model_path}"[/]...'):
            self.load_model()

    def load_model(self):
        if self.arch == "ts":
            self.model = (
                torch.jit.load(str(self.model_path.absolute())).eval().to(self.device)
            )
        else:
            state_dict = torch.load(self.model_path)

            # convert from SWA to regular model if needed
            if "n_averaged" in state_dict:
                state_dict = swa2normal(state_dict)

            if self.arch == "infer":
                if "SCPA_trunk.0.conv1_a.weight" in state_dict:
                    # pan model
                    self.arch = "pan"
                elif "model.1.sub.0.res.0.weight" in state_dict:
                    # srgan
                    self.arch = "srgan"
                elif "conv_first.weight" in state_dict:
                    # self.arch = 'mesrgan'
                    # convert msergan to esrgan
                    state_dict = mod2normal(state_dict)
                    self.arch = "esrgan"
                elif "model.0.weight" in state_dict:
                    # regular esrgan
                    self.arch = "esrgan"
                elif "CFEM.0.weight" in state_dict:
                    # ppon model
                    self.arch = "ppon"
                elif "conv_9.weight" in state_dict:
                    # wbc UNET (TODO: validate)
                    self.arch = "wbcunet"
                else:
                    raise Exception("Could not infer model parameters.")
                net_params = self.infer_params(state_dict)
            else:
                # use defaults
                net_dict = {}
                if not self.scale:
                    self.scale = 1
                if "wbcunet" in self.arch and "_tf" in self.arch:
                    self.arch = self.arch.replace("_tf", "")
                    net_dict["mode"] = "tf"
                elif "wbcunet" in self.arch:
                    net_dict["mode"] = "pt"

                net_dict["type"] = self.arch
                net_params = get_network_G_config(net_dict, self.scale)

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
        if self.arch in ("esrgan", "srgan"):
            scale2x = 0
            scalemin = 6
            n_uplayer = 0
            if self.arch == "esrgan":
                plus = False

            # TODO
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
                    if (
                        part_num > scalemin
                        and parts[0] == "model"
                        and parts[2] == "weight"
                    ):
                        # num. 2x upsample blocks
                        scale2x += 1
                    if part_num > n_uplayer:
                        # fetch out_nc from last layer shape
                        n_uplayer = part_num
                        out_nc = state_dict[block].shape[0]
                if self.arch == "esrgan":
                    if not plus and "conv1x1" in block:
                        plus = True
            nf = state_dict["model.0.weight"].shape[0]
            self.in_nc = state_dict["model.0.weight"].shape[1]
            self.out_nc = out_nc
            self.scale = 2 ** scale2x

            net_dict = {
                "type": self.arch,
                "in_nc": self.in_nc,
                "out_nc": self.out_nc,
                "nf": nf,
                "nb": nb,
            }
            if self.arch == "esrgan":
                net_dict["plus"] = plus
        elif self.arch == "wbcunet":
            self.scale = 1
            net_dict = {
                "type": self.arch,
                "mode": "pt",  # 'tf'  # TODO
                "nf": state_dict["conv.weight"].shape[0],
            }
        elif self.arch in ["ppon", "pan"]:
            # custom params inference TBD
            net_dict = {
                "type": self.arch,
                "in_nc": self.in_nc,
                "out_nc": self.out_nc,
            }

        return get_network_G_config(net_dict, self.scale)

    def chop_forward(self, data, patch_size=200, step=1.0):
        """Chop forward function used in test time.
        Converts large images into patches of size (patch_size, patch_size).
        Make sure the patch size is small enough that your GPU memory is sufficient.
        Examples: patch_size = 200 for BlindSR, 64 for ABPN
        """
        batch_size, channels, img_height, img_width = data.size()
        # if (patch_size * (1.0 - step)) % 1 < 0.5:
        #     patch_size += 1
        patch_size = min(img_height, img_width, patch_size)

        img_patches = extract_patches_2d(
            img=data,
            patch_shape=(patch_size, patch_size),
            step=[step, step],
            batch_first=True,
        ).squeeze(0)

        n_patches = img_patches.size(0)
        highres_patches = []

        with self.get_torch_ctx():
            for p in range(n_patches):
                # print(p)
                lowres_input = img_patches[p : p + 1]
                prediction = self.model(lowres_input)
                if self.arch == "ppon":
                    prediction = prediction[2]
                if self.arch == "ts":
                    # fix for CUDA out of memory.
                    prediction = prediction.detach().cpu()
                highres_patches.append(prediction)
                torch.cuda.empty_cache()

        highres_patches = torch.cat(highres_patches, 0)

        return recompose_tensor(
            highres_patches, img_height, img_width, step=step, scale=self.scale
        )

    def get_torch_ctx(self):
        if self.arch == "ts" and float(torch.__version__[:3]) < 1.8:
            # issue with torchscript: RuntimeError: CUDA driver error: a PTX JIT compilation failed
            # https://github.com/pytorch/pytorch/issues/47304
            return nullcast()
        return torch.no_grad()

    def __call__(self, data):
        if self.chop:
            t_out = self.chop_forward(
                patch_size=200,  # 100
                step=0.5,  # 0.9
                data=data,
            )
        else:
            with self.get_torch_ctx():
                t_out = self.model(data)
            if self.arch == "ppon":
                t_out = t_out[2]

        torch.cuda.empty_cache()

        return t_out


def parse_models(models_paths: str, scales_list=None):
    model_chain = (
        models_paths.split("+") if "+" in models_paths else models_paths.split(">")
    )

    all_models = get_models_paths(Path("./models"))

    full_chain = []
    for model_path in model_chain:
        full_chain.append(check_model_path(model_path, all_models))

    # try to get model scale from model name
    if not scales_list:
        scales_list = [None] * len(full_chain)
        rlt_scales = []
        for m, sc in zip(full_chain, scales_list):
            rlt_scales.append(get_scale_name(m, sc))
        scales_list = rlt_scales
    else:
        if len(scales_list) != len(model_chain):
            raise ValueError(
                f"The num. of scales {len(scales_list)} is != from number of models {len(model_chain)}"
            )

    return full_chain, scales_list


def check_model_path(model_path: str, all_models: List[Path] = None):
    # check if model exists in absolute path or ./models
    if not Path(model_path).is_file():
        model_path_a = Path("models").joinpath(model_path)
        if not model_path_a.is_file():
            # partial name search in ./models
            if all_models:
                m_list = []
                for m in all_models:
                    # if str(m).lower().find(str(model_path.lower())) >= 0:
                    if str(model_path.lower()) in str(m).lower():
                        m_list.append(m)
                if len(m_list) > 1:
                    raise ValueError(
                        f"Filter {model_path} returned multiple models: {m_list}."
                    )
                model_path = m_list[0]
            else:
                raise ValueError(f"Model {model_path} not found.")
        else:
            model_path = model_path_a
    return model_path


def get_scale_name(model_path: Path, scale=None):
    """try to get model scale from model name"""

    rlt_scale = None
    scale_name = model_path.stem[0:2].lower()
    if "x" in scale_name:
        try:
            rlt_scale = int(scale_name.replace("x", ""))
        except ValueError:
            rlt_scale = None

    if scale:
        if rlt_scale and (scale != rlt_scale):
            print(f"Warning: possible model scale mismatch on {model_path}")
        return scale
    return rlt_scale


pix2pix_extras = {
    "meval": False,  # pix2pix could produce slightly better results with eval=False (uses norm layers params)
    "strict": True,
    "normalize": True,  # pix2pix and cyclegan use normalized images
}

cyglegan_extras = {
    "meval": True,
    "strict": False,  # to ignore batch statistics that were enabled models trained with Pytorch < 0.4.0
    "normalize": True,  # pix2pix and cyclegan use normalized images
}

default_extras = {
    "meval": True,
    "strict": True,
    "normalize": False,
}


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "models",
    type=str,
    # nargs=-1,
)
@click.option(
    "-a",
    "--arch",
    type=str,
    # TODO Get a list of all available architectures
    # type=click.Choice(
    #     [
    #         "ts",
    #         "infer",
    #         "pan",
    #         "srgan",
    #         "esrgan",
    #         "ppon",
    #         "wbcunet",
    #         "unet_512",
    #         "unet_256",
    #         "unet_128",
    #         "p2p_512",
    #         "p2p_256",
    #         "p2p_128",
    #         "resnet_net",
    #         "resnet_6blocks",
    #         "resnet_6",
    #     ],
    #     case_sensitive=False,
    # ),
    required=False,
    default="infer",
    show_default=True,
    help="Model architecture.",
)
@click.option(
    "-i",
    "--input",
    type=Path,
    required=False,
    default=Path("./input"),
    show_default=True,
    help="Path to read input images.",
)
@click.option(
    "-o",
    "--output",
    type=Path,
    required=False,
    default=Path("./output"),
    show_default=True,
    help="Path to save output images.",
)
@click.option(
    "-s",
    "--scale",
    type=int,
    required=False,
    default=-1,
    help="Model scaling factor.",
)
@click.option(
    "-cf",
    "--color-fix",
    "color_correction",
    is_flag=True,
    help="Use color correction if enabled.",
)
@click.option(
    "--comp",
    is_flag=True,
    help="Save as comparison images if enabled.",
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Run in CPU if enabled.",
)
@click.option("--fp16", is_flag=True, help="Enable fp16 mode if needed.")
@click.option(
    "--norm",
    is_flag=True,
    help="Normalizes images in range [-1,1] if set, else [0,1].",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    # count=True,
    # help="Verbosity level (-v, -vv ,-vvv)",
)
def image(
    models: str,
    arch: str,
    input: Path,
    output: Path,
    scale: int,
    color_correction: bool,
    comp: bool,
    cpu: bool,
    fp16: bool,
    norm: bool,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
    )
    log = logging.getLogger()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    gpu = (
        not cpu
    )  # TODO: fp16 error with cpu: RuntimeError: "unfolded2d_copy" not implemented for 'Half'

    # TODO: all these options should be configurable
    if arch == "ts":
        # TODO: not working with torchscript unless model was traced with fp16
        fp16 = False  # True
    else:
        fp16 = fp16 and gpu

    use_guided_filter = False
    use_modcrop = False
    if "unet_" in arch or "p2p_" in arch:
        defaults = pix2pix_extras
        chop = False  # tmp, could chop to unet size
        if "512" in arch:
            resize = 512
        elif "256" in arch:
            resize = 256
        elif "128" in arch:
            resize = 128
    elif "resnet_" in arch or "cg_" in arch:
        defaults = cyglegan_extras
        chop = True
        resize = False
    elif "wbc" in arch or "wbc" in models:
        if "tf" in arch or "tf" in models:
            arch = "wbcunet_tf"
        else:
            arch = "wbcunet"
        defaults = pix2pix_extras
        chop = False  # True
        resize = False
        use_guided_filter = True
        use_modcrop = True
    else:
        defaults = default_extras
        resize = False
        chop = True

    meval = defaults["meval"]
    strict = defaults["strict"]
    normalize = defaults["normalize"] or norm

    if fp16:
        torch.set_default_tensor_type(
            torch.cuda.HalfTensor if gpu else torch.HalfTensor
        )
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and gpu
        else torch.device("cpu")
    )

    cf = color_correction  # color fix
    comp = comp  # save comparison images
    model_path = models
    output_dir = output
    # TODO: chain scales
    scale = scale if scale != -1 else None
    # TODO: chain archs

    model_chain, scale_chain = parse_models(model_path)

    models = []
    for mc, sc in zip(model_chain, scale_chain):
        models.append(
            Model(mc, arch, sc, device=device, meval=meval, strict=strict, chop=chop)
        )

    try:
        images = get_images_paths(input)
    except AssertionError as e:
        log.error(e)
        sys.exit(1)

    with Progress(
        # SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task_upscaling = progress.add_task("Upscaling", total=len(images))
        for image_path in images:
            log.info(f'Upscaling "{image_path.name}"')
            img = read_img(image_path)

            # if not isinstance(img, np.ndarray):
            if img is None:
                log.warning(f'Error reading image "{image_path}", skipping.')
                progress.advance(task_upscaling)
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
            save_img_path = output_dir.joinpath(f"{image_path.stem}.png")
            if comp:
                save_img_comp([img, img_out], save_img_path)
            else:
                save_img(img_out, save_img_path)
            progress.advance(task_upscaling)


@cli.command()
def video():
    pass


if __name__ == "__main__":
    cli()
