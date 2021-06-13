# iNNfer
Simple inference codes for AI models.

This is a companion repository to [BasicSR](https://github.com/victorca25/BasicSR), in order to more easily produce results with models trained with it.

Currently, the model architectures supported are for: Super-Resolution, Restoration (denoise, deblur) and image to image translation. Support for the remaining architectures (SRFlow, Video, etc) is planned.

## Features

Below is a (non-comprehensive) list of features currently available in the project. More are planned (see below).

-   Support for all super-resolution and restoration models available in BasicSR, like: `ESRGAN` (`RRDB`, both original and modified architectures), `SRGAN` (`SRResNet`), `PPON`, and `PAN`. (`SRFlow` is pending).
-   "Chop forward" option, to automatically divide large images to smaller crops to prevent `CUDA` errors due to exhausted `VRAM`.
-   Automatic inference of model scale, either from the model name (for example: `4x_PPON_CGP4.pth` will be interpreted as scale `4`) or from the network configuration (currently only for `ESRGAN`, others planned). If the scale cannot be infered, you can use the scale flag with the scale factor, like: `-scale 4`.
-   Automatic inference of model architecture (for super-resolution and restoration at the moment, others planned). Mening that, for example, `ESRGAN`, `PPON` and `PAN` models can be chained with no additional requirement. The exact architecture can also be provided, specially if not using a default network configuration.
-   Model chaining, to pass the input images through a sequence of models.
-   Direct support for Stochastic Weight Averaging (SWA) models. Will automatically be converted to a regular model.
-   Support for image to image translation models: `pix2pix` (`UNet`) and `CycleGAN` (`ResNet`).
-   Support for `TorchScript` models. Use flag `-arch ts` (can not be chained yet).
-   Automatic color correction, for models that modify the color hues when applied. Use flag `-cf`.
-   Use of `fp16` format to reduce memory requirements. Technically, this operates with less accuracy than the default `fp32`, but for all tests so far, the errors where imperceptible. If you suspect any issue, fp16 can be disabled with the `-no_fp16` flag.
-   Runs on NVidia GPUs if available by default or on CPU. Can also force to CPU with the flag `-cpu`.
-   Partial model name support. You don't need to use the models' full names, only part of the name that identifies it as different from others.
-   Option to do a side by side comparison of the input images to the output results with the `-comp` flag.


## Planned features

-   On the fly model interpolation, with automatic model compatibility detection. (Except for TorchScript models)
-   Option to use the Consistency Enforcement Module ([CEM](https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/CEM/README.md)).
-   Additional color correction alternatives.
-   Photograph restoration pipeline.
-   Add Colab Notebook.


## Example simple usage

You need to provide a directory where the `input` images to be processed are located and an `output` where the results will be saved. By default, these directories will be `./input/` and `./output/` respectively, but you can modify those with the `-input` and `-output` flags.

If you obtain a [trained](https://github.com/victorca25/BasicSR/blob/master/docs/pretrained.md) model, either the original from a paper or from the [model database](https://upscale.wiki/wiki/Model_Database), you can place it in the `./models/` directory.

As an example, if you want to use the `Fatality` model from the database, you will download the model (`4x_Fatality_01_265000_G.pth`) and move it to `./models/`.

Once there and with the input images ready, you can run obtain the results simply by running:

```bash
python run.py -m fatal
```

And the results will be saved in `./output/`.

## More cases

### Model chaining

To chain multiple models, you need to provide a sequence of model names to the `-m` flag. For example, to first remove JPEG artifacts and then upscale images, you can fetch one of the JPEG denoising models from the database (Example: `1x_JPEG_60-80.pth`) and an upscaling model (Example: `4x_Fatality_01_265000_G.pth`) and use a plus sign (`+`) between their names.

```bash
python run.py -m jpeg+fatal
```

Note that there's technically no limit to how many models can be chained, but if the models are for upscaling, iamge sizes can become impossible to manage in memory. This is mostly a hardware limitation. You can also chain the same model multiple times to the images, which can produce interesting results in some cases.

### Image to image to image translation

For these cases, for now you'll need to provide the network architecture used to train the model. For example, from the [trained](https://github.com/victorca25/BasicSR/blob/master/docs/pretrained.md) model available for `pix2pix` and `CycleGAN`, that will correspond to `unet_256` and `resnet_9blocks` for the `CycleGAN` case.

For example, to try out the `label2facade` model (`facades_label2photo.pth`), you need to run:

```bash
python run.py -m facade -a unet_256
```

This will produce a single result:

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/121805922-bbc91e00-cc4d-11eb-8961-accdd7eb4269.png" height="200">
</p>

For a side by side comparison between input and output, add the `-comp` flag:

```bash
python run.py -m facade -a unet_256 -comp
```

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/121805956-e915cc00-cc4d-11eb-9d34-cb7683ad6b5f.png" height="200">
</p>

Similarly, to test the `ukiyoe` CycleGAN model (either `photo2ukiyoe.pth` or `style_ukiyoe.pth`), with a comparison run:

```bash
python run.py -m facade -a unet_256 -comp
```

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/121806039-5c1f4280-cc4e-11eb-8377-62c907039e3a.png" height="200">
</p>

### TorchScript models

TorchScript models are directly supported, just be aware that these models need to be run in the same fashion they were traced. For example, if the option for using GPUs was used when they were created, they will only be able to run in NVidia GPUs with CUDA support. [Here](https://mega.nz/folder/34gBAKaL#TH9d7HKzYxpu1lgC2X9H_Q) you will find a number of models from the model database that were already converted to TorchScript (using GPU, CPU versions can be made available if needed) and are ready to use.

For example, to use the `4xRealSR_DF2K_JPEG.pt` model, just execute:

```bash
python run.py -m realsr
```

One of the advantages these TorchScript models have is that they no longer require explicit support of the network architecture, so you could use any model of any architecture that has been converted to TorchScript with this code, even if the architecture is not supported. This is useful if you need to use other features like the color correction.


### Color correction option

Some models introduce color changes that may not be desired. For that reason, there are options that can be used to correct those changes.

Using an example image from the [Manga109](http://www.manga109.org/ja/index.html) set, with a model that intentionally introduces heavy color changes, run:

```bash
python run.py -m shin -comp
```

And this produces this result:

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/121806509-62aeb980-cc50-11eb-8f99-1cfe64ff446a.png" height="200">
</p>


To try to fix the colors, only add the color fix flag `-cf`, like:

```bash
python run.py -m shin -comp -cf
```

And you will obtain a version of the upscale that more closely matches the colors of the original image:

<p align="center">
   <img src="https://user-images.githubusercontent.com/41912303/121806581-b91bf800-cc50-11eb-9193-d688472339d3.png" height="200">
</p>

This flag will work, even if multiple models are chained.
