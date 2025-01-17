import os
import subprocess

import tyro


def extract_frames(
    video_path: str,
    output_path: str,
    video_nr: int,
    res: int,
    ext: str,
    skip_time: int = 1,
    start_time: str = "00:00:00",
    end_time: str | None = None,
):
    # get video name
    video_name = os.path.basename(video_path)
    # remove extension
    video_name = os.path.splitext(video_name)[0]
    #
    video_id = format(video_nr, "02d")
    os.makedirs(output_path, exist_ok=True)
    to_str = f"-to {end_time}" if end_time else ""
    # get video height and width
    h_str = subprocess.check_output(
        f"ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 {video_path}",
        shell=True,
    )
    h = int(h_str)
    height = h // res
    command = f"ffmpeg -i {video_path} -vf \"select='not(mod(n,{skip_time}))',scale=-1:{height}\" -vsync vfr -ss {start_time} {to_str} {output_path}/{video_id}_%05d.{ext}"
    subprocess.call(command, shell=True)


# def main(
#     video_path: str,
#     output_path: str,
#     video_nr: int = 0,
#     res: int = 1,  # 1x
#     ext: str = "jpg",
#     skip_time: int = 1,
#     start_time: str = "00:00:00",
#     end_time: str | None = None,
# ):
#     print(f"Extracting frames from {video_path}")
#     print(f"Output root: {output_path}")
#     print(f"Res: {res}")
#     print(f"Extension: {ext}")
#     print(f"Skip time: {skip_time}")
#     print(f"Start time: {start_time}")
#     print(f"End time: {end_time}")
        
#     extract_frames(
#         video_path, output_path, video_nr, res, ext, skip_time, start_time, end_time
#     )


# if __name__ == "__main__":
#     tyro.cli(main)
