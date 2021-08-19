import logging
import sys
from pathlib import Path
from typing import List
from utils.utils import get_images_paths, read_img, save_img, save_img_comp
from upscale import Upscale

import click
from rich import get_console, print
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TimeRemainingColumn,
)
from rich.traceback import install as install_traceback

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
        for image_path in images:
            log.info(f'Upscaling "{image_path.name}"')
            img = read_img(image_path)

            # if not isinstance(img, np.ndarray):
            if img is None:
                log.warning(f'Error reading image "{image_path}", skipping.')
                progress.advance(task_upscaling)
                continue

            img_out = upscale.image(img, color_correction)

            # save images
            save_img_path = output.joinpath(
                image_path.parent.relative_to(input)
            ).joinpath(f"{image_path.stem}.png")

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
