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


LARGE_SCALE_MULTIPLIER = 0.05
SCALE_MULTIPLIER = 0.05


def get_scale(scene_radius: float) -> float:
    scale = SCALE_MULTIPLIER
    if scene_radius <= 1.0:
        return scale
    else:
        return scale + (scene_radius * LARGE_SCALE_MULTIPLIER)


class GUI:

    def __init__(
        self,
        mv_data: MVDataset,
        config: dict = {}
    ):
        # 
        self.scale = get_scale(mv_data.get_scene_radius())
        
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.window_w, self.window_h = config["width"], config["height"]
        self.window = gui.Application.instance.create_window(
            "MVDatasets", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.init_widget()

        # dict to hold frustums
        self.frustum_dict = {}

        # iterate over training cameras and add them
        for i, camera in enumerate(mv_data.get_split("train")):
            self.add_camera(camera, f"train_{i}", color=[0, 0, 0])

        # iterate over test cameras and add them
        for i, camera in enumerate(mv_data.get_split("test")):
            self.add_camera(camera, f"test_{i}", color=[0, 1, 0])

        # iterate over validation cameras and add them
        for i, camera in enumerate(mv_data.get_split("val")):
            self.add_camera(camera, f"val_{i}", color=[0, 0, 1])

        # iterate over point clouds and add them
        self.pc_dict = {}
        for i, pc in enumerate(mv_data.get_point_clouds()):
            self.add_point_cloud(pc, f"pc_{i}")

        app.run()

    def add_camera(self, camera: Camera, name: str, color=[0, 0, 0]):

        from experimental.gui.frustum import create_frustum

        width, height = camera.get_resolution()
        intrinsics = camera.get_intrinsics()
        frustum = create_frustum(width, height, intrinsics, color)
        self.frustum_dict[name] = frustum
        self.widget3d.scene.add_geometry(name, frustum.line_set, self.material_line)
        #
        c2w = camera.get_pose()
        self.widget3d.scene.set_geometry_transform(name, c2w.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)

        return frustum

    def add_point_cloud(self, pc: PointCloud, name: str):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points_3d)
        if pc.points_rgb is not None:
            # squeeze the color values
            pc.points_rgb = pc.points_rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(pc.points_rgb)
        self.pc_dict[name] = pcd
        self.widget3d.scene.add_geometry(name, pcd, self.material)
        self.widget3d.scene.show_geometry(name, self.pc_chbox.checked)

        return pcd
    
    def init_widget(self):

        #
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        #
        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)
        #
        self.window.add_child(self.widget3d)
        #
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        #
        self.material_line = rendering.MaterialRecord()
        self.material_line.shader = "unlitLine"

        # instantiate axis
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5 * self.scale, origin=[0, 0, 0]
        )
        self.widget3d.scene.add_geometry("axis", self.axis, self.material)

        #
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())

        #
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        # Viewpoint Options ================================================
        # self.panel.add_child(gui.Label("Viewpoint Options"))

        # viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        # vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        # vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))

        # ## Check boxes
        # vp_subtile1.add_child(gui.Label("Camera follow options"))
        # chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        # self.followcam_chbox = gui.Checkbox("Follow Camera")
        # self.followcam_chbox.checked = True
        # chbox_tile.add_child(self.followcam_chbox)

        # self.staybehind_chbox = gui.Checkbox("From Behind")
        # self.staybehind_chbox.checked = True
        # chbox_tile.add_child(self.staybehind_chbox)
        # vp_subtile1.add_child(chbox_tile)

        # viewpoint_tile.add_child(vp_subtile1)
        # viewpoint_tile.add_child(vp_subtile2)
        # self.panel.add_child(viewpoint_tile)

        # 3D Objects ======================================================
        self.panel.add_child(gui.Label("3D Objects"))

        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.axis_chbox = gui.Checkbox("Axis")
        self.axis_chbox.checked = True
        self.axis_chbox.set_on_checked(self._on_axis_chbox)
        chbox_tile_3dobj.add_child(self.axis_chbox)

        self.pc_chbox = gui.Checkbox("Point Clouds")
        self.pc_chbox.checked = True
        self.pc_chbox.set_on_checked(self._on_pc_chbox)
        chbox_tile_3dobj.add_child(self.pc_chbox)

        #
        self.panel.add_child(chbox_tile_3dobj)

        # # screenshot buttom
        # self.screenshot_btn = gui.Button("Screenshot")
        # self.screenshot_btn.set_on_clicked(
        #     self._on_screenshot_btn
        # )  # set the callback function
        # self.panel.add_child(self.screenshot_btn)

        # Rendering Tab ===================================================
        # tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        # tabs = gui.TabControl()

        # tab_info = gui.Vert(0, tab_margins)
        # self.output_info = gui.Label("Number of Gaussians: ")
        # tab_info.add_child(self.output_info)

        # self.in_rgb_widget = gui.ImageWidget()
        # self.in_depth_widget = gui.ImageWidget()
        # tab_info.add_child(gui.Label("Input Color/Depth"))
        # tab_info.add_child(self.in_rgb_widget)
        # tab_info.add_child(self.in_depth_widget)

        # tabs.add_tab("Info", tab_info)
        # self.panel.add_child(tabs)

        # add panel to window
        self.window.add_child(self.panel)

    # def init_glfw(self):
    #     window_name = "headless rendering"

    #     if not glfw.init():
    #         exit(1)

    #     glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    #     window = glfw.create_window(
    #         self.window_w, self.window_h, window_name, None, None
    #     )
    #     glfw.make_context_current(window)
    #     glfw.swap_interval(0)
    #     if not window:
    #         glfw.terminate()
    #         exit(1)
    #     return window

    # def render_o3d_image(self, render):

    #     rgb = (
    #         (torch.clamp(render, min=0, max=1.0) * 255)
    #         .byte()
    #         .permute(1, 2, 0)  # HWC
    #         .contiguous()
    #         .cpu()
    #         .numpy()
    #     )
    #     render_img = o3d.geometry.Image(rgb)
    #     return render_img

    # def render_gui(self):
    #     if not self.init:
    #         return

    #     render = torch.zeros((3, 256, 256))
    #     self.render_img = self.render_o3d_image(render)
    #     self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.8
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        print_log("Closing window")
        self.is_done = True
        return True  # False would cancel the close

    def _on_pc_chbox(self, is_checked):
        print_log(f"Point Clouds checked: {is_checked}")
        for name in self.pc_dict.keys():
            print_log(f"{name}")
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_cameras_chbox(self, is_checked):
        print_log(f"Cameras checked: {is_checked}")
        for name in self.frustum_dict.keys():
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_axis_chbox(self, is_checked):
        print_log(f"Axis checked: {is_checked}")
        if is_checked:
            self.widget3d.scene.remove_geometry("axis")
            self.widget3d.scene.add_geometry("axis", self.axis, self.material)
        else:
            self.widget3d.scene.remove_geometry("axis")


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
