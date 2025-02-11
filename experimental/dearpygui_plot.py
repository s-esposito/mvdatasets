import matplotlib.pyplot as plt
import numpy as np
import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg


class GUI:
    def __init__(self, width, height):

        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()

        dpg.show_viewport()

    # desctructor
    def __del__(self):
        dpg.destroy_context()

    def show(self, image, title="title"):
        # get image resolution
        width, height = image.shape[1], image.shape[0]
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width, height, image, format=dpg.mvFormat_Float_rgba, id="img"
            )
        with dpg.window(label=title):
            dpg.add_image("img")

    def run(self):
        dpg.start_dearpygui()


fig = plt.figure(figsize=(11.69, 8.26), dpi=100)
canvas = FigureCanvasAgg(fig)
ax = fig.gca()
canvas.draw()
buf = canvas.buffer_rgba()
image = np.asarray(buf)
image = image.astype(np.float32) / 255

gui = GUI(1280, 720)
gui.show(image, title="Image")
gui.run()
