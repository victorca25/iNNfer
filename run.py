import logging
import sys
from pathlib import Path
from typing import List

import click
import imageio
from imageio.plugins.ffmpeg import FfmpegFormat
from rich import get_console, print
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from rich.traceback import install as install_traceback

from upscale import Upscale
from utils.utils import get_images_paths, read_img, save_img, save_img_comp

install_traceback()


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
    "-se",
    "--skip-existing",
    is_flag=True,
    help="Skip existing output files.",
)
@click.option(
    "-di",
    "--delete-input",
    is_flag=True,
    help="Delete input files after upscaling.",
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
    skip_existing: bool,
    delete_input: bool,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
    )
    log = logging.getLogger()

    upscale = Upscale(models, arch, scale, cpu, fp16, norm)

    try:
        images = get_images_paths(input)
    except AssertionError as e:
        log.error(e)
        sys.exit(1)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task_upscaling = progress.add_task("Upscaling", total=len(images))
        for img_path in images:
            log.info(f'Upscaling "{img_path.relative_to(input)}"')

            save_img_path = output.joinpath(
                img_path.parent.relative_to(input)
            ).joinpath(f"{img_path.stem}.png")

            if skip_existing and save_img_path.is_file():
                log.warning(
                    f'Image "{save_img_path.relative_to(output)}" already exists, skipping.'
                )
                if delete_input:
                    img_path.unlink(missing_ok=True)
                progress.advance(task_upscaling)
                continue

            img = read_img(img_path)

            # if not isinstance(img, np.ndarray):
            if img is None:
                log.warning(
                    f'Error reading image "{img_path.relative_to(input)}", skipping.'
                )
                progress.advance(task_upscaling)
                continue

            img_out = upscale.image(img, color_correction)

            # save images
            if comp:
                save_img_comp([img, img_out], save_img_path)
            else:
                save_img(img_out, save_img_path)

            if delete_input:
                img_path.unlink(missing_ok=True)

            progress.advance(task_upscaling)


@cli.command()
@click.argument(
    "models",
    type=str,
)
@click.option(
    "-a",
    "--arch",
    type=str,
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
    default=Path("./input/video.mp4"),
    show_default=True,
    help="Path to read input video.",
)
@click.option(
    "-o",
    "--output",
    type=Path,
    required=False,
    default=Path("./output/video.mp4"),
    show_default=True,
    help="Path to save output video.",
)
@click.option(
    "-s",
    "--scale",
    type=int,
    required=False,
    default=-1,
    help="Model scaling factor.",
)
@click.option("--fp16", is_flag=True, help="Enable fp16 mode if needed.")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
)
def video(
    models: str,
    arch: str,
    input: Path,
    output: Path,
    scale: int,
    fp16: bool,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
    )
    log = logging.getLogger()

    input = input.resolve()
    output = output.resolve()
    if not input.exists():
        log.error(f'Input video "{input}" does not exist.')
        sys.exit(1)
    elif input.is_dir():
        log.error(f'Input video "{input}" is a directory.')
        sys.exit(1)
    elif output.is_dir():
        log.error(f'Output video "{output}" is a directory.')
        sys.exit(1)

    upscale = Upscale(models, arch, scale, fp16=fp16)

    video_reader: FfmpegFormat.Reader = imageio.get_reader(str(input.absolute()))
    fps = video_reader.get_meta_data()["fps"]
    num_frames = video_reader.count_frames()

    scenes = []
    last_scene_frame = 0
    for i in range(250, num_frames, 250):
        scenes.append((last_scene_frame + 1, i))
        last_scene_frame = i
    if len(scenes) == 0:
        scenes.append((1, num_frames))
    log.info(f"Video divided into {len(scenes)} scenes.")

    print("TODO")


if __name__ == "__main__":
    cli()
