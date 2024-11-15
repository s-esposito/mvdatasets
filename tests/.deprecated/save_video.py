import imageio

out_dir = "./videos"
mp4_name = "test.mp4"

writer = imageio.get_writer(
    f"{out_dir}/{mp4_name}", mode="I", fps=30, codec="libx264", bitrate="16M"
)
