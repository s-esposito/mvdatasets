import os
import numpy as np
from PIL import Image
from mvdatasets.utils.geometry import qvec2rotmat
from mvdatasets.utils.images import image_to_numpy
from mvdatasets.utils.printing import print_error


def read_points3D(reconstruction):
    point_cloud = []
    for point3D_id, point3D in reconstruction.points3D.items():
        point_cloud.append(point3D.xyz)
    point_cloud = np.array(point_cloud)
    return point_cloud


def read_cameras_params(reconstruction):

    cameras_params = {}
    for camera_id, camera in reconstruction.cameras.items():
        params = {}
        # PINHOLE
        if camera.model_id == 1:
            params["fx"] = camera.params[0]
            params["fy"] = camera.params[1]  # fy
            params["cx"] = camera.params[2]  # cx
            params["cy"] = camera.params[3]  # cy
        # # SIMPLE_RADIAL
        # elif camera.model_id == 2:
        #     intrinsics[0, 0] = camera.params[0]  # fx
        #     intrinsics[1, 1] = camera.params[0]  # fy = fx
        #     intrinsics[0, 2] = camera.params[1]  # cx
        #     intrinsics[1, 2] = camera.params[2]  # cy
        #     # camera.params[3]  # k1
        else:
            print_error(f"camera model {camera.model_id} not implemented.")
            exit(1)
        cameras_params[str(camera_id)] = params
    return cameras_params


def read_cameras(reconstruction, images_path):

    cameras_params = read_cameras_params(reconstruction)

    cameras_meta = []
    for image_id, image in reconstruction.images.items():

        rotation = qvec2rotmat(image.qvec)
        translation = image.tvec

        params = cameras_params[str(image.camera_id)]

        # load PIL image
        img_pil = Image.open(os.path.join(images_path, image.name))
        img_np = image_to_numpy(img_pil, use_uint8=True)

        cameras_meta.append(
            {
                "id": image_id,
                "rotation": rotation,
                "translation": translation,
                "params": params,
                "img": img_np,
            }
        )

    # order by "id"
    cameras_meta = sorted(cameras_meta, key=lambda x: x["id"])

    return cameras_meta
