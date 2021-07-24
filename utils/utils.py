import os.path as osp
from os import walk as osw

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .colors import *

try:
    import rawpy
    rawpy_available = True
except ImportError:
    rawpy_available = False


MODEL_EXTENSIONS = ['.pth', '.pt']

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp',
                    'tga', '.tif', '.tiff', '.dng']

MAX_VALUES_BY_DTYPE = {
    np.dtype("int8"): 127,
    np.dtype("uint8"): 255,
    np.dtype("int16"): 32767,
    np.dtype("uint16"): 65535,
    np.dtype("int32"): 2147483647,
    np.dtype("uint32"): 4294967295,
    np.dtype("int64"): 9223372036854775807,
    np.dtype("uint64"): 18446744073709551615,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
}


def is_ext_file(filename, extensions=IMG_EXTENSIONS):
    return any(filename.endswith(extension) for extension in extensions)


def scan_dir(path, extensions=IMG_EXTENSIONS):
    if not osp.isdir(path):
        raise AssertionError(f'{path:s} is not a valid directory')
    files_list = []
    for dirpath, _, fnames in sorted(osw(path)):
        for fname in sorted(fnames):
            if is_ext_file(fname, extensions):
                img_path = osp.join(dirpath, fname)
                files_list.append(img_path)
    return files_list


def get_models_paths(path):
    """ Get model path list from model folder"""
    models = scan_dir(path, MODEL_EXTENSIONS)
    if not models:
        raise AssertionError(f'{path:s} has no valid model file')
    return models


def get_images_paths(path):
    """ Get image path list from image folder"""
    images = scan_dir(path, IMG_EXTENSIONS)
    if not images:
        raise AssertionError(f'{path:s} has no valid image file')
    return images


def read_img(path=None):
    """ Reads an image using cv2 (or rawpy if dng)
    Arguments:
        path: image path to read
    Output:
        Numpy HWC, BGR, [0,255] by default 
    """

    img = None
    if path:
        if rawpy_available and path[-3:].lower() == 'dng':
            # if image is a DNG
            with rawpy.imread(path) as raw:
                img = raw.postprocess()
        else:
            # if image can be read by cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raise AssertionError("Empty path provided.")
    return img



def save_img(img, img_path, mode='RGB', scale=None):
    """ Save a single image to the defined path """
    if scale:
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(img_path, img)


def merge_imgs(img_list):
    """ Auxiliary function to horizontally concatenate images in
        a list using cv2.hconcat
    """
    if isinstance(img_list, list):
        img_h = 0
        img_v = 0
        for img in img_list:
            if img.shape[0] > img_v:
                img_h = img.shape[0]
            if img.shape[1] > img_v:
                img_v = img.shape[1]

        img_list_res = []
        for img in img_list:
            if img.shape[1] < img_v or img.shape[0] < img_h:
                img_res = cv2.resize(img, (img_v, img_h), interpolation=cv2.INTER_NEAREST)
                img_list_res.append(img_res)
            else:
                img_list_res.append(img)
        
        return cv2.hconcat(img_list_res)
    elif isinstance(img_list, np.ndarray):
        return img_list
    else:
        raise NotImplementedError('To merge images img_list should be a list of cv2 images.')


def save_img_comp(img_list, img_path, mode='RGB'):
    """ Create a side by side comparison of multiple images in a list
        to save to a defined path
    """
    # lr_resized = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # comparison = cv2.hconcat([lr_resized, sr_img])
    comparison = merge_imgs(img_list)
    save_img(img=comparison, img_path=img_path, mode=mode)


def denorm(x, min_max=(-1.0, 1.0)):
    """ Denormalize from [-1,1] range to [0,1]
        formula: xi' = (xi - mu)/sigma
        Example: "out = (x + 1.0) / 2.0" for denorm 
            range (-1,1) to (0,1)
        for use with proper act in Generator output (ie. tanh)
    """
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    if isinstance(x, torch.Tensor):
        return out.clamp(0, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError(
            "Got unexpected object type, expected torch.Tensor or np.ndarray")

def norm(x): 
    """ Normalize (z-norm) from [0,1] range to [-1,1] """
    out = (x - 0.5) * 2.0
    if isinstance(x, torch.Tensor):
        return out.clamp(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, -1, 1)
    else:
        raise TypeError(
            "Got unexpected object type, expected torch.Tensor or np.ndarray")


def np2tensor(img, bgr2rgb=True, data_range=1., normalize=False,
        change_range=True, add_batch=True):
    """ Converts a numpy image array into a Tensor array.
    Parameters:
        img (numpy array): the input image numpy array
        add_batch (bool): choose if new tensor needs batch dimension added
    """
    if not isinstance(img, np.ndarray): #images expected to be uint8 -> 255
        raise TypeError("Got unexpected object type, expected np.ndarray")

    # check how many channels the image has, then condition. ie. RGB, RGBA, Gray
    # if bgr2rgb:
    #     img = img[:, :, [2, 1, 0]] #BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        t_dtype = np.dtype("float32")
        img = img.astype(t_dtype)/maxval  # ie: uint8 = /255

    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() #"HWC to CHW" and "numpy to tensor"
    if bgr2rgb:
        # BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.)
        if img.shape[0] % 3 == 0:  # RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
            img = bgr_to_rgb(img)
        elif img.shape[0] == 4:  # RGBA
            img = bgra_to_rgba(img)
    if add_batch:
        img.unsqueeze_(0)  # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
    if normalize:
        img = norm(img)
    return img


def tensor2np(img, rgb2bgr=True, remove_batch=True, data_range=255, 
              denormalize=False, change_range=True, imtype=np.uint8):
    """ Converts a Tensor array into a numpy image array.
    Parameters:
        img (tensor): the input image tensor array
            4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        remove_batch (bool): choose if tensor of shape BCHW needs to be squeezed 
        denormalize (bool): Used to denormalize from [-1,1] range back to [0,1]
        imtype (type): the desired type of the converted numpy array (np.uint8 
            default)
    Output: 
        img (np array): 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("Got unexpected object type, expected torch.Tensor")
    n_dim = img.dim()

    # TODO: Check: could denormalize here in tensor form instead, but end result is the same
    
    img = img.float().cpu()  
    
    if n_dim in (4, 3):
        # if n_dim == 4, has to convert to 3 dimensions
        if n_dim == 4 and remove_batch:
            # remove a fake batch dimension
            img = img.squeeze(dim=0)
        
        if img.shape[0] == 3 and rgb2bgr:  # RGB
            # RGB to BGR -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgb_to_bgr(img).numpy()
        elif img.shape[0] == 4 and rgb2bgr:  # RGBA
            # RGBA to BGRA -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgba_to_bgra(img).numpy()
        else:
            img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(
            f'Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim:d}')

    # if rgb2bgr:
        #img_np = img_np[[2, 1, 0], :, :] #RGB to BGR -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    # TODO: Check: could denormalize in the begining in tensor form instead
    if denormalize:
        img_np = denorm(img_np)  # denormalize if needed
    if change_range:
        img_np = np.clip(data_range*img_np, 0, data_range).round()  # clip to the data_range
        
    # has to be in range (0,255) before changing to np.uint8, else np.float32
    return img_np.astype(imtype)


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def linear_resize(img, st=256):
    h, w =  img.shape[0:2]
    if not (h % st == 0) or not (w % st == 0):
        oh = -(-h // st) * st
        ow = -(-w // st) * st
        linear = srgb2linear(img)
        linear = cv2.resize(linear, dsize=(ow, oh), interpolation=cv2.INTER_CUBIC)
        img = linear2srgb(linear)
    return img


def color_fix(imgA, imgB):
    """ Fix coloration changes by adding the difference in
        the low frequency of the original image
    """
    kernel_size = 3
    scaling = False

    # convert images to linear space
    imgA = srgb2linear(imgA)
    imgB = srgb2linear(imgB)

    # downscale imgB to imgA size if needed
    hA, wA =  imgA.shape[0:2]
    hB, wB =  imgB.shape[0:2]
    if hA < hB and wA < wB:
        scaling = True
        imgB_ds = cv2.resize(
            imgB, dsize=(wA, hA), interpolation=cv2.INTER_CUBIC)
    else:
        imgB_ds = imgB

    # compute the difference (ie: LR - SR)
    diff = imgA - imgB_ds
    
    # gaussian blur for low frequency information (colors)
    #TODO: test with guided filter
    blurred = cv2.GaussianBlur(diff, (kernel_size, kernel_size), 0)
    
    # upscale if needed and add diff back to the imgB
    if scaling:
        blurred = cv2.resize(
            blurred, dsize=(wB, hB), interpolation=cv2.INTER_CUBIC)
    rlt = blurred + imgB

    # rlt = denorm(rlt, min_max=(rlt.min(), rlt.max()))

    # back to srgb space and return
    return linear2srgb(rlt)
    

def extract_patches_2d(img, patch_shape, step=None, batch_first=False):
    """ Convert a 4D tensor into a 5D tensor of patches (crops) of
    the original tensor. Uses unfold to extract sliding local blocks
    from an batched input tensor.
    Arguments:
        img: the image batch to crop
        patch_shape: tuple with the shape of the last two dimensions (H,W) 
            after crop
        step: the size of the step used to slide the blocks in each dimension.
            If each value 0.0 < step < 1.0, the overlap will be relative to the
            patch size * step
        batch_first: return tensor with batch as the first dimension or the 
            second
    Reference: 
    https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
    """
    if step is None: step = [1.0, 1.0]
    patch_H, patch_W = patch_shape[0], patch_shape[1]

    # pad to fit patch dimensions
    if(img.size(2) < patch_H):
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if(img.size(3) < patch_W):
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)

    # steps to overlay crops of the images
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if(isinstance(step[1], float)) else step[1]

    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,
                                    img[:, :, -patch_H:, :].permute(0, 1, 3, 2).unsqueeze(2)),dim=2)

    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,
                                     patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)

    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)

    if(batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches


def recompose_tensor(patches, height, width, step=None, scale=1):
    """ Reconstruct images that have been cropped to patches.
    Unlike reconstruct_from_patches_2d(), this function allows to 
    use blending between the patches if they were generated a 
    step between 0.5 (50% overlap) and 1.0 (0% overlap), 
    relative to the original patch size
    Arguments:
        patches: the image patches
        height: the original image height
        width: the original image width
        step: the overlap step factor, from 0.5 to 1.0
        scale: the scale at which the patches are in relation to the 
            original image
    References:
    https://github.com/sunreef/BlindSR/blob/master/src/image_utils.py
    https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78

    """
    if step is None: step = [1.0, 1.0]
    assert isinstance(step, float) and step >= 0.5 and step <= 1.0

    full_height = scale * height
    full_width = scale * width
    batch_size, channels, patch_size, _ = patches.size()
    overlap = scale * int(round((1.0 - step) * (patch_size / scale)))

    effective_patch_size = int(step * patch_size)
    patch_H, patch_W = patches.size(2), patches.size(3)
    img_size = (patches.size(0), patches.size(1), max(full_height, patch_H), max(full_width, patch_W))

    step = [step, step]
    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0])
    step_int[1] = int(patch_W * step[1])

    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    n_patches_height = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    n_patches_width = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol

    final_batch_size = batch_size // (n_patches_height * n_patches_width)

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()

    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor


def normalize_kernel2d(x: torch.Tensor) -> torch.Tensor:
    """Normalizes kernel."""
    if len(x.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(x.size()))
    norm: torch.Tensor = x.abs().sum(dim=-1).sum(dim=-1)
    return x / (norm.unsqueeze(-1).unsqueeze(-1))


def compute_padding(kernel_size):
    """ Computes padding tuple. For square kernels, pad can be an
    int, else, a tuple with an element for each dimension.
    """
    # 4 or 6 ints:  (padding_left, padding_right, padding_top, padding_bottom)
    if isinstance(kernel_size, tuple):
        kernel_size = list(kernel_size)

    if isinstance(kernel_size, int):
        return kernel_size//2
    elif isinstance(kernel_size, list):
        computed = [k // 2 for k in kernel_size]

        out_padding = []

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
            # for even kernels we need to do asymetric padding
            if kernel_size[i] % 2 == 0:
                padding = computed_tmp - 1
            else:
                padding = computed_tmp
            out_padding.append(padding)
            out_padding.append(computed_tmp)
        return out_padding


def filter2D(x: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect', dim: int =2,
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.
    The function applies a given kernel to a tensor. The kernel
    is applied independently at each depth channel of the tensor.
    Before applying the kernel, the function applies padding
    according to the specified mode so that the output remains
    in the same shape.
    Args:
        x: the input tensor with shape of :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input tensor.
            The kernel shape must be :math:`(1, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized: If True, kernel will be L1 normalized.
    Return:
        the convolved tensor of same size and numbers of channels
            as the input.
    """

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = x.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(x.device).to(x.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(x, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape

    tmp_kernel = tmp_kernel.expand(c, -1, -1, -1)

    # convolve the tensor with the kernel.
    if dim == 1:
        conv = F.conv1d
    elif dim == 2:
        conv = F.conv2d
    elif dim == 3:
        conv = F.conv3d
    else:
        raise RuntimeError(
            f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

    return conv(input_pad, tmp_kernel, groups=c, padding=0, stride=1)


def get_box_kernel(kernel_size: int = 5, dim=2):
    if isinstance(kernel_size,  (int, float)):
        kernel_size = [kernel_size] * dim

    kx = kernel_size[0]
    ky = kernel_size[1]
    box_kernel = torch.Tensor(np.ones((kx, ky)) / (float(kx)*float(ky)))

    return box_kernel


def guided_filter(x: torch.Tensor, y: torch.Tensor,
    x_HR: torch.Tensor = None, ks=None, r=None, eps:float=1e-2,
    box_kernel=None, mode:str='regular', conv_a=None) -> torch.Tensor:
    """ Guided filter / FastGuidedFilter function.
    This is a kind of edge-preserving smoothing filter that can
    filter out noise or texture while retaining sharp edges. One
    key assumption of the guided filter is that the relation
    between guidance x and the filtering output is linear.
    Arguments:
        x: guidance image with shape [b, c, h, w].
        y: filtering input image with shape [b, c, h, w].
        x_HR: optional high resolution guidance map for joint
            upsampling (for 'fast' or 'conv' modes).
        ks (int): kernel size for the box/mean filter. In reference to
            the window radius "r": kx = ky = ks = (2*r)+1
        r (int): optional radius for the window. Can use instead of ks.
        box_kernel (tensor): precalculated box_kernel (optional).
        mode: select between the guided filter types: 'regular',
            'fast' or 'conv' (convolutional).
        conv_a (nn.Sequential): the convolutional layers to use for
            'conv' mode to calculate the 'A' parameter.
        eps: regularization ε, penalizing large A values.
            eps = 1e-8 in the original paper.
    Returns:
        output: filtered image
    """

    if not isinstance(box_kernel, torch.Tensor):
    # get the box_kernel if not provided
        if not ks:
            if r:
                ks = (2*r)+1
            else:
                raise ValueError("Either kernel size (ks) or radius (r) "
                                 "for the window are required.")

        # mean filter. The window size is defined by the kernel size.
        box_kernel = get_box_kernel(kernel_size = ks)

    x_shape = x.shape
    # y_shape = y.shape
    if isinstance(x_HR, torch.Tensor):
        x_HR_shape = x_HR.shape

    box_kernel = box_kernel.to(x.device)
    N = filter2D(torch.ones((1, 1, x_shape[-2], x_shape[-1])),
        box_kernel).to(x.device)

    # note: similar to SSIM calculation
    mean_x = filter2D(x, box_kernel) / N
    mean_y = filter2D(y, box_kernel) / N
    cov_xy = (filter2D(x*y, box_kernel) / N) - mean_x*mean_y
    var_x = (filter2D(x*x, box_kernel) / N) - mean_x*mean_x

    # linear coefficients A, b
    if mode == 'conv':
        A = conv_a(torch.cat([cov_xy, var_x], dim=1))
    else:
        # regular or fast GuidedFilter
        A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x  # according to original GF paper, needs to add: "+ x"

    # mean_A; mean_b
    if mode == 'fast' or mode == 'conv':
        mean_A = F.interpolate(
            A, (x_HR_shape[-2], x_HR_shape[-1]),
            mode='bilinear', align_corners=True)
        mean_b = F.interpolate(
            b, (x_HR_shape[-2], x_HR_shape[-1]),
            mode='bilinear', align_corners=True)
        output = mean_A * x_HR + mean_b
    else:
        # regular GuidedFilter
        mean_A = filter2D(A, box_kernel) / N
        mean_b = filter2D(b, box_kernel) / N
        output = mean_A * x + mean_b

    return output


def normal2mod(state_dict):
    if 'model.0.weight' in state_dict:
        print('Converting and loading an RRDB model to modified RRDB')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        crt_net['conv_first.weight'] = state_dict['model.0.weight']
        crt_net['conv_first.bias'] = state_dict['model.0.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('model.1.sub.', 'RRDB_trunk.')
                if '.0.weight' in k:
                    ori_k = ori_k.replace('.0.weight', '.weight')
                elif '.0.bias' in k:
                    ori_k = ori_k.replace('.0.bias', '.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['trunk_conv.weight'] = state_dict['model.1.sub.23.weight']
        crt_net['trunk_conv.bias'] = state_dict['model.1.sub.23.bias']
        crt_net['upconv1.weight'] = state_dict['model.3.weight']
        crt_net['upconv1.bias'] = state_dict['model.3.bias']
        crt_net['upconv2.weight'] = state_dict['model.6.weight']
        crt_net['upconv2.bias'] = state_dict['model.6.bias']
        crt_net['HRconv.weight'] = state_dict['model.8.weight']
        crt_net['HRconv.bias'] = state_dict['model.8.bias']
        crt_net['conv_last.weight'] = state_dict['model.10.weight']
        crt_net['conv_last.bias'] = state_dict['model.10.bias']
        state_dict = crt_net

    return state_dict


def mod2normal(state_dict):
    if 'conv_first.weight' in state_dict:
        print('Converting and loading a modified RRDB model to normal RRDB')
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict


def swa2normal(state_dict):
    if 'n_averaged' in state_dict:
        print('Attempting to convert a SWA model to a regular model\n')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        for k in items.copy():
            if 'n_averaged' in k:
                print('n_averaged: {}'.format(state_dict[k]))
            elif 'module.module.' in k:
                ori_k = k.replace('module.module.', '')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        state_dict = crt_net

    return state_dict
