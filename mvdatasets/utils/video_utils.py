import os
import subprocess
from pathlib import Path


def extract_frames(
    video_path: Path,
    output_path: Path = None,
    subsample_factor: int = 1,
    ext: str = "png",
    skip_time: int = 1,
    start_time: str = "00:00:00",
    end_time: str | None = None,
):
    # get video name
    video_name = os.path.basename(video_path)
    # remove extension
    video_name = os.path.splitext(video_name)[0]
    # if outout path is not given
    if output_path is None:
        # create output folder in same folder containing the video
        # whose name is the video name
        output_path = video_path.parent / video_name
    else:
        # create output folder in output path
        # whose name is the video name
        output_path = output_path / video_name
    # create output folder
    os.makedirs(output_path, exist_ok=True)
    to_str = f"-to {end_time}" if end_time else ""
    # get video height and width
    h_str = subprocess.check_output(
        f"ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 {video_path}",
        shell=True,
    )
    h = int(h_str)
    height = h // subsample_factor
    command = f"ffmpeg -i {video_path} -vf \"select='not(mod(n,{skip_time}))',scale=-1:{height}\" -vsync vfr -ss {start_time} {to_str} {output_path}/%05d.{ext}"
    subprocess.call(command, shell=True)
