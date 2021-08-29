import logging
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Generator, List, Optional, Tuple

# import imageio
import numpy as np
import torch
from decord import cpu  # , gpu
from decord import VideoReader
from imageio.plugins.ffmpeg import FfmpegFormat
from rich import get_console

from architectures import get_network
from utils.defaults import get_network_G_config
from utils.utils import (
    are_same_imgs,
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
        in_nc: int = 3,
        out_nc: int = 3,
        device: str = "cpu",
        meval: bool = True,
        strict: bool = True,
        chop: bool = True,
    ) -> None:
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

    def load_model(self) -> None:
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

    def infer_params(self, state_dict) -> dict:
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

    def chop_forward(self, data, patch_size: int = 200, step: float = 1.0):
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


def parse_models(
    models_paths: str, scales_list=None
) -> Tuple[List[str], List[Optional[int]]]:
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


def check_model_path(model_path: str, all_models: List[Path] = None) -> Path:
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
    return Path(model_path)


def get_scale_name(model_path: Path, scale=None) -> Optional[int]:
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
            logging.getLogger().warning(
                f"Possible model scale mismatch on {model_path}"
            )
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


class ModelDevice:
    device: torch.device
    locks: List[Lock]
    models: List[Model]

    def __init__(self, device: torch.device, processed_by_device: int = 1) -> None:
        self.device = device
        self.locks = [Lock() for _ in range(processed_by_device)]
        self.models = []

    @property
    def name(self):
        if self.device.type == "cuda":
            device_name = torch.cuda.get_device_name(self.device.index)
        else:
            device_name = "CPU"
        return device_name


class Process:
    def __init__(
        self,
        models_str: str,
        arch: str,
        scale: int,
        cpu: bool = False,
        fp16: bool = False,
        device_id: int = 0,
        multi_gpu: bool = False,
        processed_by_device: int = 1,
        normalize: bool = False,
    ) -> None:
        self.models_str = models_str
        self.arch = arch
        self.scale = scale
        self.cpu = cpu
        self.fp16 = fp16
        self.multi_gpu = multi_gpu
        self.normalize = normalize
        self.log = logging.getLogger()

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        self.gpu = (
            not cpu
        )  # TODO: fp16 error with cpu: RuntimeError: "unfolded2d_copy" not implemented for 'Half'

        # TODO: all these options should be configurable
        if self.arch == "ts":
            # TODO: not working with torchscript unless model was traced with fp16
            self.fp16 = False  # True
        else:
            self.fp16 = self.fp16 and self.gpu

        self.use_guided_filter = False
        self.use_modcrop = False
        if "unet_" in self.arch or "p2p_" in self.arch:
            defaults = pix2pix_extras
            self.chop = False  # tmp, could chop to unet size
            if "512" in self.arch:
                self.resize = 512
            elif "256" in self.arch:
                self.resize = 256
            elif "128" in self.arch:
                self.resize = 128
        elif "resnet_" in self.arch or "cg_" in self.arch:
            defaults = cyglegan_extras
            self.chop = True
            self.resize = False
        elif "wbc" in self.arch or "wbc" in self.models_str:
            if "tf" in self.arch or "tf" in self.models_str:
                self.arch = "wbcunet_tf"
            else:
                self.arch = "wbcunet"
            defaults = pix2pix_extras
            self.chop = False  # True
            self.resize = False
            self.use_guided_filter = True
            self.use_modcrop = True
        else:
            defaults = default_extras
            self.resize = False
            self.chop = True

        self.meval = defaults["meval"]
        self.strict = defaults["strict"]
        self.normalize = defaults["normalize"] or self.normalize

        if self.fp16:
            torch.set_default_tensor_type(
                torch.cuda.HalfTensor if self.gpu else torch.HalfTensor
            )
        self.model_devices: List[ModelDevice] = []
        if self.multi_gpu:
            for i in range(torch.cuda.device_count()):
                self.model_devices.append(
                    ModelDevice(
                        torch.device(f"cuda:{i}"),
                        processed_by_device=processed_by_device,
                    )
                )
            # Uncomment to use the cpu
            # self.model_devices.append(
            #     ModelDevice(
            #         torch.device("cpu"), processed_by_device=processed_by_device
            #     )
            # )
        else:
            self.model_devices.append(
                ModelDevice(
                    torch.device(f"cuda:{device_id}")
                    if torch.cuda.is_available() and self.gpu
                    else torch.device("cpu"),
                    processed_by_device=processed_by_device,
                )
            )

        # TODO: chain scales
        self.scale = self.scale if self.scale != -1 else None
        # TODO: chain archs
        self.load_models()

    def load_models(self) -> None:
        model_chain, scale_chain = parse_models(self.models_str)
        for model_device in self.model_devices:
            model_device.models = []
            for mc, sc in zip(model_chain, scale_chain):
                model_device.models.append(
                    Model(
                        mc,
                        self.arch,
                        sc,
                        device=model_device.device,
                        meval=self.meval,
                        strict=self.strict,
                        chop=self.chop,
                    )
                )

    def get_available_model_device(
        self, sleep_time: float = 0.25, first_lock: bool = True
    ) -> Tuple[torch.device, int]:
        model_device: ModelDevice = None
        while model_device == None:
            for md in self.model_devices:
                num_lock = 0
                if first_lock:
                    lock = md.locks[0]
                else:
                    lock = None
                    for n in range(len(md.locks)):
                        if not md.locks[n].locked():
                            lock = md.locks[n]
                            break
                        num_lock += 1
                if lock != None and not lock.locked():
                    model_device = md
                    lock.acquire()
                    break
            if model_device == None:
                # self.log.warning(f"No GPU available. Waiting...")
                time.sleep(sleep_time)
        return model_device, num_lock

    def image(
        self,
        img: np.ndarray,
        color_correction: bool = False,
        device: torch.device = None,
        multi_gpu_release_device: bool = True,
    ) -> np.ndarray:
        # TODO: can pad|resize|crop images to next size accepted by network
        if self.resize:
            img = linear_resize(img, self.resize)

        if self.use_modcrop:
            img = modcrop(img, 4)

        if device == None:
            if self.multi_gpu:
                model_device, _ = self.get_available_model_device()
            else:
                model_device = self.model_devices[0]
        else:
            model_device = next(md for md in self.model_devices if md.device == device)

        t_img = np2tensor(img, normalize=self.normalize).to(model_device.device)
        t_img = t_img.half() if self.fp16 else t_img

        t_out = t_img.clone()
        for model in model_device.models:
            t_out = model(t_out)
            if self.use_guided_filter:
                # note: r can be configured here to control details in results
                t_out = guided_filter(t_img, t_out, r=1, eps=5e-3)

        img_out = tensor2np(t_out.detach(), denormalize=self.normalize)

        if self.multi_gpu and multi_gpu_release_device:
            model_device.locks[0].release()

        if color_correction:
            img_out = color_fix(img, img_out)

        return img_out

    def video(
        self,
        video_path: Path,
        start_frame: int = 0,
        end_frame: int = None,
        ssim: bool = False,
        min_ssim: float = 0.9987,
        deinterpaint: str = None,
        device: torch.device = None,
    ) -> Generator[Optional[np.ndarray], None, None]:
        # video_reader: FfmpegFormat.Reader = imageio.get_reader(
        #     str(video_path.absolute())
        # )
        # num_frames = video_reader.count_frames()
        # video_reader.set_image_index(start_frame)
        video_reader = VideoReader(
            str(video_path.absolute()), ctx=cpu(0)
        )  # TODO decord gpu
        num_frames = len(video_reader)
        video_reader.seek(start_frame)
        end_frame = end_frame or num_frames

        last_frame = None
        last_frame_ai = None
        for frame_idx in range(start_frame, end_frame):
            # frame = video_reader.get_next_data()
            frame = video_reader.next().asnumpy()
            if deinterpaint is not None:
                for i in range(0 if deinterpaint == "even" else 1, frame.shape[0], 2):
                    frame[i : i + 1] = (0, 255, 0)  # (B, G, R)

            if last_frame is not None and are_same_imgs(
                last_frame, frame, ssim, min_ssim
            ):
                frame_ai = last_frame_ai
            else:
                frame_ai = self.image(
                    frame, device=device, multi_gpu_release_device=False
                )
            last_frame = frame
            last_frame_ai = frame_ai
            yield frame_ai
