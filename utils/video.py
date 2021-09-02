import logging
from typing import List, Tuple

from scenedetect import FrameTimecode, SceneManager, VideoManager
from scenedetect.detectors import ContentDetector


def find_scenes(
    video_path: str, threshold: float = 30.0
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path], logger=logging.getLogger())
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()
