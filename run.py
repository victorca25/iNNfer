import configparser
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from threading import Lock, Thread
from typing import List, Tuple

import click
import numpy as np
import torch
from humanize.time import precisedelta
from rich import get_console, print
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.progress import BarColumn, Progress, TaskID, TimeRemainingColumn
from rich.traceback import install as install_traceback

try:
    import imageio
    from imageio.plugins.ffmpeg import FfmpegFormat

    imageio_available = True
except ImportError:
    imageio_available = False

from model import ModelDevice, Process
from utils.utils import (
    are_same_imgs,
    get_images_paths,
    read_img,
    save_img,
    save_img_comp,
)

install_traceback()


@click.group()
def cli():
    pass


def image_thread_func(
    img: np.ndarray,
    device: torch.device,
    color_correction: bool,
    comp: bool,
    save_img_path: Path,
    process: Process,
    progress: Progress,
    task_processing: TaskID,
):
    img_out = process.image(img, color_correction=color_correction, device=device)

    # save images
    if comp:
        save_img_comp([img, img_out], save_img_path)
    else:
        save_img(img_out, save_img_path)

    progress.advance(task_processing)


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
    "-did",
    "--device-id",
    type=int,
    required=False,
    default=0,
    help="The numerical ID of the GPU you want to use.",
)
@click.option("-mg", "--multi-gpu", is_flag=True, help="Multi GPU.")
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
    help="Delete input files after processing.",
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
    device_id: int,
    multi_gpu: bool,
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

    process = Process(
        models,
        arch,
        scale,
        cpu,
        fp16=fp16,
        device_id=device_id,
        multi_gpu=multi_gpu,
        normalize=norm,
    )

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
        task_processing = progress.add_task("Processing", total=len(images))
        threads = []
        for img_path in images:
            save_img_path = output.joinpath(
                img_path.parent.relative_to(input)
            ).joinpath(f"{img_path.stem}.png")

            if skip_existing and save_img_path.is_file():
                log.warning(
                    f'Image "{save_img_path.relative_to(output)}" already exists, skipping.'
                )
                if delete_input:
                    img_path.unlink(missing_ok=True)
                progress.advance(task_processing)
                continue

            img = read_img(img_path)

            # if not isinstance(img, np.ndarray):
            if img is None:
                log.warning(
                    f'Error reading image "{img_path.relative_to(input)}", skipping.'
                )
                progress.advance(task_processing)
                continue
            if multi_gpu:
                model_device, num_lock = process.get_available_model_device()
            else:
                model_device = process.model_devices[0]
            log.info(
                f'Processing "{img_path.relative_to(input)}"'
                + (
                    f' using "{model_device.name}"'
                    if multi_gpu and len(process.model_devices) > 1
                    else ""
                )
            )

            if multi_gpu:
                x = Thread(
                    target=image_thread_func,
                    args=(
                        img,
                        model_device.device,
                        color_correction,
                        comp,
                        save_img_path,
                        process,
                        progress,
                        task_processing,
                    ),
                )
                threads.append(x)
                x.start()
            else:
                image_thread_func(
                    img,
                    model_device.device,
                    color_correction,
                    comp,
                    save_img_path,
                    process,
                    progress,
                    task_processing,
                )

            if delete_input:
                img_path.unlink(missing_ok=True)

        for thread in threads:
            thread.join()


def video_thread_func(
    model_device: ModelDevice,
    num_lock: int,
    multi_gpu: bool,
    input: Path,
    output: Path,
    start_frame: int,
    end_frame: int,
    num_frames: int,
    progress: Progress,
    task_processed_id: TaskID,
    ai_processed_path: Path,
    fps: int,
    quality: float,
    ffmpeg_params: str,
    deinterpaint: str,
    ssim: bool,
    min_ssim: float,
    process: Process,
    config: configparser.ConfigParser,
    scenes_ini: Path,
):
    log = logging.getLogger()
    start_time = time.process_time()
    start_frame_str = str(start_frame).zfill(len(str(num_frames)))
    end_frame_str = str(end_frame).zfill(len(str(num_frames)))
    task_scene_desc = f'Scene [green]"{start_frame_str}_{end_frame_str}"[/]'
    if multi_gpu and len(process.model_devices) > 1:
        task_scene_desc += f" ({model_device.name})"
    task_scene_id = progress.add_task(
        description=task_scene_desc,
        total=end_frame - start_frame,
        completed=0,
        refresh=True,
    )
    video_writer_params = {"quality": quality, "macro_block_size": None}
    if ffmpeg_params:
        if "-crf" in ffmpeg_params:
            del video_writer_params["quality"]
        video_writer_params["output_params"] = ffmpeg_params.split()
    if output.suffix == ".gif":
        del video_writer_params["quality"]
        del video_writer_params["macro_block_size"]
    video_writer: FfmpegFormat.Writer = imageio.get_writer(
        str(
            ai_processed_path.joinpath(
                f"{start_frame_str}_{end_frame_str}{output.suffix}"
            ).absolute()
        ),
        fps=fps,
        **video_writer_params,
    )
    duplicated_frames = 0
    total_duplicated_frames = 0
    last_frame_ai = None
    start_duplicated_frame = 0
    for frame_idx, frame_ai in enumerate(
        process.video(
            input,
            start_frame - 1,
            end_frame,
            ssim,
            min_ssim,
            deinterpaint=deinterpaint,
            device=model_device.device,
        )
    ):
        current_frame_idx = start_frame + frame_idx
        video_writer.append_data(frame_ai)
        if last_frame_ai is not None:
            if (last_frame_ai == frame_ai).all():
                if duplicated_frames == 0:
                    start_duplicated_frame = current_frame_idx
                duplicated_frames += 1
            else:
                if duplicated_frames != 0:
                    start_duplicated_frame_str = str(start_duplicated_frame).zfill(
                        len(str(num_frames))
                    )
                    end_duplicated_frame_str = str(current_frame_idx - 1).zfill(
                        len(str(num_frames))
                    )
                    log.info(
                        f"Detected {duplicated_frames} duplicated frame{'' if duplicated_frames==1 else 's'} ({start_duplicated_frame_str}{'' if duplicated_frames==1 else '-' + end_duplicated_frame_str})"
                    )
                    total_duplicated_frames += duplicated_frames
                    duplicated_frames = 0

        last_frame_ai = frame_ai
        progress.advance(task_processed_id)
        progress.advance(task_scene_id)
    if duplicated_frames != 0:
        start_duplicated_frame_str = str(start_duplicated_frame).zfill(
            len(str(num_frames))
        )
        end_duplicated_frame_str = str(end_frame).zfill(len(str(num_frames)))
        log.info(
            f"Detected {duplicated_frames} duplicated frame{'' if duplicated_frames==1 else 's'} ({start_duplicated_frame_str}{'' if duplicated_frames==1 else '-' + end_duplicated_frame_str})"
        )
        total_duplicated_frames += duplicated_frames
        duplicated_frames = 0

    video_writer.close()
    task_scene = next(task for task in progress.tasks if task.id == task_scene_id)

    config.set(f"{start_frame_str}_{end_frame_str}", "processed", "True")
    config.set(
        f"{start_frame_str}_{end_frame_str}",
        "duplicated_frames",
        f"{total_duplicated_frames}",
    )
    config.set(
        f"{start_frame_str}_{end_frame_str}",
        "average_fps",
        f"{task_scene.finished_speed:.2f}",
    )
    with open(scenes_ini, "w") as configfile:
        config.write(configfile)
    log.info(
        f"Frames from {str(start_frame).zfill(len(str(num_frames)))} to {str(end_frame).zfill(len(str(num_frames)))} processed in {precisedelta(dt.timedelta(seconds=time.process_time() - start_time))}"
    )
    if total_duplicated_frames > 0:
        total_frames = end_frame - (start_frame - 1)
        seconds_saved = (
            (
                (1 / task_scene.finished_speed * total_frames)
                - (total_duplicated_frames * 0.04)  # 0.04 seconds per duplicate frame
            )
            / (total_frames - total_duplicated_frames)
            * total_duplicated_frames
        )
        log.info(
            f"Total number of duplicated frames from {str(start_frame).zfill(len(str(num_frames)))} to {str(end_frame).zfill(len(str(num_frames)))}: {total_duplicated_frames} (saved ≈ {precisedelta(dt.timedelta(seconds=seconds_saved))})"
        )
    progress.remove_task(task_scene_id)
    if multi_gpu:
        model_device.locks[num_lock].release()


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
    "-did",
    "--device-id",
    type=int,
    required=False,
    default=0,
    help="The numerical ID of the GPU you want to use.",
)
@click.option("-mg", "--multi-gpu", is_flag=True, help="Multi GPU.")
@click.option(
    "-spg",
    "--scenes-per-gpu",
    type=int,
    required=False,
    default=1,
    help="Number of scenes to be upscaled at the same time using the same GPU.",
)
@click.option(
    "-q",
    "--quality",
    type=click.FloatRange(1, 10),
    required=False,
    default=10,
    help="Video quality.",
)
@click.option(
    "-ffmpeg",
    "--ffmpeg-params",
    type=str,
    required=False,
    help='FFmpeg parameters to save the scenes. If -crf is present, the quality parameter will be ignored. Example: "-c:v libx265 -crf 5 -pix_fmt yuv444p10le -preset medium -x265-params pools=none -threads 8".',
)
@click.option(
    "--ssim",
    is_flag=True,
    help="True to enable duplication frame removal using ssim. False to use np.all().",
)
@click.option(
    "-ms",
    "--min-ssim",
    type=float,
    required=False,
    default=0.9987,
    help="Min SSIM value.",
)
@click.option(
    "-dp",
    "--deinterpaint",
    type=click.Choice(["even", "odd"], case_sensitive=False),
    required=False,
    help="De-interlacing by in-painting. Fills odd or even rows with green (#00FF00). Useful for models like Joey's 1x_DeInterPaint.",
)
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
    device_id: int,
    multi_gpu: bool,
    scenes_per_gpu: bool,
    quality: float,
    ffmpeg_params: str,
    ssim: bool,
    min_ssim: float,
    deinterpaint: str,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
    )
    log = logging.getLogger()

    if not imageio_available:
        raise Exception(
            "Video processing requires imageio and imageio-ffmpeg packages."
        )

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

    process = Process(
        models,
        arch,
        scale,
        fp16=fp16,
        device_id=device_id,
        multi_gpu=multi_gpu,
        processed_by_device=scenes_per_gpu,
    )

    video_reader: FfmpegFormat.Reader = imageio.get_reader(str(input.absolute()))
    fps = video_reader.get_meta_data()["fps"]
    num_frames = video_reader.count_frames()
    video_reader.close()

    project_path = output.parent.joinpath(f"{output.stem}").absolute()
    ai_processed_path = project_path.joinpath("scenes")
    scenes_ini = project_path.joinpath("scenes.ini")
    frames_todo: List[Tuple[int, int]] = []
    frames_processed: List[Tuple[int, int]] = []
    config = configparser.ConfigParser()
    if project_path.is_dir():
        resume_mode = True
        log.info(f'Resuming project "{project_path}"')
        config.read(scenes_ini)
        for scene in config.sections():
            start_frame, end_frame = scene.split("_")
            if config.getboolean(scene, "processed") == True:
                frames_processed.append((int(start_frame), int(end_frame)))
            else:
                frames_todo.append((int(start_frame), int(end_frame)))
    else:
        resume_mode = False
        scenes = []
        with get_console().status("Dividing the video into scenes..."):
            last_scene_frame = 0
            for i in range(250, num_frames, 250):
                scenes.append((last_scene_frame + 1, i))
                last_scene_frame = i
            if len(scenes) == 0:
                scenes.append((1, num_frames))
            if last_scene_frame != num_frames:
                scenes.append((last_scene_frame + 1, num_frames))

        log.info(
            f"Video divided into {len(scenes)} scene{'' if len(scenes)==1 else 's'}."
        )

        ai_processed_path.mkdir(parents=True, exist_ok=True)
        if num_frames != scenes[-1][1]:
            log.error("num_frames != scenes[-1][1]")
        for scene in scenes:
            start_frame = str(scene[0]).zfill(len(str(num_frames)))
            end_frame = str(scene[1]).zfill(len(str(num_frames)))
            config[f"{start_frame}_{end_frame}"] = {
                "processed": "False",
                "duplicated_frames": "None",
                "average_fps": "None",
            }
            frames_todo.append((int(start_frame), int(end_frame)))

        with open(scenes_ini, "w") as configfile:
            config.write(configfile)

    with Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(),
        TimeRemainingColumn(),
        # FpsSpeedColumn(),
    ) as progress:
        num_frames_processed = 0
        for start_frame, end_frame in frames_processed:
            num_frames_processed += end_frame - start_frame + 1
        task_processed_id = progress.add_task(
            f'Processing [green]"{input.name}"[/]', total=num_frames
        )
        if num_frames_processed > 0:
            log.info(f"Skipped {num_frames_processed} frames already processed")
            progress.update(
                task_processed_id, completed=num_frames_processed, refresh=True
            )
        threads = []
        for start_frame, end_frame in frames_todo:
            num_lock = 0
            if multi_gpu:
                model_device, num_lock = process.get_available_model_device(
                    first_lock=False
                )
            else:
                model_device = process.model_devices[0]
            if multi_gpu:
                x = Thread(
                    target=video_thread_func,
                    args=(
                        model_device,
                        num_lock,
                        multi_gpu,
                        input,
                        output,
                        start_frame,
                        end_frame,
                        num_frames,
                        progress,
                        task_processed_id,
                        ai_processed_path,
                        fps,
                        quality,
                        ffmpeg_params,
                        deinterpaint,
                        ssim,
                        min_ssim,
                        process,
                        config,
                        scenes_ini,
                    ),
                )
                threads.append(x)
                x.start()
            else:
                video_thread_func(
                    model_device,
                    num_lock,
                    multi_gpu,
                    input,
                    output,
                    start_frame,
                    end_frame,
                    num_frames,
                    progress,
                    task_processed_id,
                    ai_processed_path,
                    fps,
                    quality,
                    ffmpeg_params,
                    deinterpaint,
                    ssim,
                    min_ssim,
                    process,
                    config,
                    scenes_ini,
                )
        for thread in threads:
            thread.join()

    with open(
        project_path.joinpath("scene_list.txt"), "w", encoding="utf-8"
    ) as outfile:
        for video_path in ai_processed_path.glob(f"*{output.suffix}"):
            outfile.write(f"file '{video_path.relative_to(project_path).as_posix()}'\n")
    total_duplicated_frames = 0
    total_average_fps = 0
    for section in config.sections():
        total_duplicated_frames += config.getint(section, "duplicated_frames")
        total_average_fps += config.getfloat(section, "average_fps")
    total_average_fps = total_average_fps / len(config.sections())
    if not resume_mode:
        task_processed = next(
            task for task in progress.tasks if task.id == task_processed_id
        )
        total_average_fps = task_processed.finished_speed
    if total_duplicated_frames > 0:
        seconds_saved = (
            (
                (1 / total_average_fps * num_frames)
                - (total_duplicated_frames * 0.04)  # 0.04 seconds per duplicate frame
            )
            / (num_frames - total_duplicated_frames)
            * total_duplicated_frames
        )
        log.info(
            f"Total number of duplicated frames: {total_duplicated_frames} (saved ≈ {precisedelta(dt.timedelta(seconds=seconds_saved))})"
        )
    log.info(f"Total FPS: {total_average_fps:.2f}")
    print("\nProcessed completed!\n")

    bad_scenes = []
    with get_console().status(
        f"Checking the correct number of frames of the {output.suffix} files..."
    ):
        for video_path in ai_processed_path.glob(f"*{output.suffix}"):
            start_frame, end_frame = video_path.stem.split("_")
            num_frames = int(end_frame) - int(start_frame) + 1
            with imageio.get_reader(str(video_path.absolute())) as video_reader:
                frames_mp4 = video_reader.count_frames()
            if num_frames != frames_mp4:
                bad_scenes.append(f"{video_path.stem}")

    if len(bad_scenes) > 0:
        for scene in bad_scenes:
            config.set(scene, "processed", "False")
        with open(scenes_ini, "w") as configfile:
            config.write(configfile)
        if len(bad_scenes) == 1:
            bad_scenes_str = f"[green]{bad_scenes[0]}[/]"
        else:
            bad_scenes_str = f'[green]{"[/], [green]".join(bad_scenes[:-1])}[/] and [green]{bad_scenes[-1]}[/]'
        print(f"The following scenes were incorrectly processed: {bad_scenes_str}.")
        print(f"Please re-run the script to finish processing them.")
    else:
        print(
            f'Go to the "{project_path}" directory and run the following command to concatenate the scenes.'
        )
        print(
            Markdown(
                f"`ffmpeg -f concat -safe 0 -i scene_list.txt -c copy {output.name}`"
            )
        )


if __name__ == "__main__":
    cli()
