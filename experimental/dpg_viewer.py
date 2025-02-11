import sys
import tyro
import numpy as np
import torch
import time
import open3d as o3d
from pathlib import Path
from typing import List
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# import glfw
# from OpenGL import GL as gl
from mvdatasets import Camera
from mvdatasets.geometry.primitives import PointCloud
from mvdatasets import MVDataset
from mvdatasets.utils.printing import print_warning, print_log
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler

#
import dearpygui.dearpygui as dpg
from dataclasses import dataclass


class Open3DRenderer:
    def __init__(self, width, height):
        
        # 
        self.height = height
        self.width = width
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.width, self.height
        )

        # setup basic scene
        self.renderer.scene.set_background([1, 1, 1, 1])  # Black background

        # add a simple 3D object (a coordinate frame)
        # instantiate axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        material = o3d.visualization.rendering.MaterialRecord()
        self.renderer.scene.add_geometry("axes", axis, material)

        # setup camera
        bounds = self.renderer.scene.bounding_box
        min_bound = np.array(bounds.get_min_bound(), np.float32)
        max_bound = np.array(bounds.get_max_bound(), np.float32)
        # get max norm of the bounding box
        max_norm = np.linalg.norm(max_bound - min_bound)
        center = np.array(bounds.get_center(), np.float32)  # Look-at point

        # use vertical fov to setup the camera
        vertical_fov = 60.0  # Field of view in degrees
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        eye = np.array([0.0, 0.0, max_norm], dtype=np.float32)  # Camera position
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Up direction

        # setup the camera with near and far clip planes
        self.renderer.setup_camera(
            vertical_fov, center, eye, up, near_clip=0.1, far_clip=100.0
        )

        # # use and instrinics matrix to setup the camera
        # intrinsic_matrix = np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]], dtype=np.float64)
        # extrinsic_matrix = np.eye(4)  # No transformation

        # # setup camera using explicit intrinsic and extrinsic matrices
        # self.renderer.setup_camera(intrinsic_matrix, extrinsic_matrix, 800, 600)
        
        #
        self.frustum_dict = {}
        self.pc_dict = {}
        
    def add_camera(self, camera: Camera, name: str, color=[0, 0, 0]):

        from experimental.gui.frustum import create_frustum

        width, height = camera.get_resolution()
        intrinsics = camera.get_intrinsics()
        frustum = create_frustum(width, height, intrinsics, color)
        self.frustum_dict[name] = frustum
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "unlitLine"
        self.renderer.scene.add_geometry(name, frustum.line_set, material)
        #
        c2w = camera.get_pose()
        self.renderer.scene.set_geometry_transform(name, c2w.astype(np.float64))
        # self.renderer.scene.show_geometry(name, self.cameras_chbox.checked)

        return frustum

    def add_point_cloud(self, pc: PointCloud, name: str):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points_3d)
        if pc.points_rgb is not None:
            # squeeze the color values
            pc.points_rgb = pc.points_rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(pc.points_rgb)
        self.pc_dict[name] = pcd
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "unlitLine"
        self.renderer.scene.add_geometry(name, pcd, material)
        # self.renderer.scene.show_geometry(name, self.pc_chbox.checked)

        return pcd

    def render_to_numpy(self):
        """Renders the Open3D scene to a numpy array. Results in [0, 255],"""
        image = self.renderer.render_to_image()
        img_data = np.asarray(image)  # Convert to NumPy array
        return img_data


class GUI:
    def __init__(self, mv_data: MVDataset, config: dict = {}):
        #
        self._running = False

        #
        self.padding = 10

        # create Open3D renderer
        self.o3d_renderer = Open3DRenderer(
            width=config["width"] - 2 * self.padding,
            height=config["height"] - 2 * self.padding,
        )

        dpg.create_context()
        dpg.create_viewport(
            title="MVDatasets", width=config["width"], height=config["height"]
        )
        dpg.setup_dearpygui()

        with dpg.texture_registry(show=False):
            # add the texture
            dpg.add_raw_texture(
                self.o3d_renderer.width,
                self.o3d_renderer.height,
                np.zeros(
                    (self.o3d_renderer.width, self.o3d_renderer.height, 4),
                    dtype=np.float32,
                ),
                tag="canvas",
                format=dpg.mvFormat_Float_rgba,
            )

        with dpg.window(tag="primary_window"):
            dpg.add_image("canvas")
            
        # set the privary window
        dpg.set_primary_window("primary_window", True)
        
        # 
        # iterate over training cameras and add them
        for i, camera in enumerate(mv_data.get_split("train")):
            self.o3d_renderer.add_camera(camera, f"train_{i}", color=[0, 0, 0])

        # iterate over test cameras and add them
        for i, camera in enumerate(mv_data.get_split("test")):
            self.o3d_renderer.add_camera(camera, f"test_{i}", color=[0, 1, 0])

        # iterate over validation cameras and add them
        for i, camera in enumerate(mv_data.get_split("val")):
            self.o3d_renderer.add_camera(camera, f"val_{i}", color=[0, 0, 1])
            
        # iterate over point clouds and add them
        for i, pc in enumerate(mv_data.get_point_clouds()):
            self.o3d_renderer.add_point_cloud(pc, f"pc_{i}")

    def __del__(self):
        # destroy context
        dpg.destroy_context()

    def run(self):

        # show the viewport
        dpg.show_viewport()

        self._running = True

        while dpg.is_dearpygui_running() and self._running:

            # start_time = time.time()

            self.draw()

            dpg.render_dearpygui_frame()

            # Yield CPU time to the background thread
            time.sleep(0.01)

            # end_time = time.time()
            # elapsed_time = end_time - start_time

        print_log("gui rendering loop stopped.")

    def draw(self):
        """Updates the Dear PyGui texture with Open3D rendering"""
        o3d_img = self.o3d_renderer.render_to_numpy()  # uint8
        # convert to float32
        o3d_img = o3d_img.astype(np.float32) / 255.0

        # Flip vertically (Open3D has top-left origin, DPG expects bottom-left)
        # img_data = np.flip(img_data, axis=0)
        # print(img_data.shape)
        
        # 
        img_data = o3d_img
        
        # convert RGB to RGBA (DPG requires 4 channels)
        if img_data.shape[-1] == 3:
            # add Alpha
            img_data = np.dstack(
                (
                    img_data,
                    np.full(
                        (img_data.shape[0], img_data.shape[1], 1), 1.0, dtype=np.float32
                    ),
                )
            )

        # Update the texture
        dpg.set_value("canvas", img_data)

        time.sleep(1 / 60)  # 60 FPS


def main(cfg: ExampleConfig, pc_paths: List[Path]):

    # device = cfg.machine.device
    datasets_path = cfg.datasets_path
    # output_path = cfg.output_path
    scene_name = cfg.scene_name
    dataset_name = cfg.data.dataset_name

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        config=cfg.data.asdict(),
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    gui_config = {"width": 1920, "height": 1080}

    app = GUI(mv_data, config=gui_config)
    app.run()


if __name__ == "__main__":

    # python experimental/open3d_viewer.py --datasets-path /home/stefano/Data data:nerf-synthetic --data.pose-only --data.max-cameras-distance 1

    # custom exception handler
    sys.excepthook = custom_exception_handler

    # parse arguments
    args = tyro.cli(ExampleConfig)

    # get test preset
    test_preset = get_dataset_test_preset(args.data.dataset_name)
    # scene name
    if args.scene_name is None:
        args.scene_name = test_preset["scene_name"]
        print_warning(
            f"scene_name is None, using preset test scene {args.scene_name} for dataset"
        )
    # additional point clouds paths (if any)
    pc_paths = test_preset["pc_paths"]

    # start the example program
    main(args, pc_paths)
