import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import product, combinations

from datasets.utils.camera import Camera

def plot_cameras(cameras, angle=60):

    '''
    # cameras: list of Camera objects

    Use in a notebook with:
    
    import ipywidgets as widgets
    from IPython.display import display
    @widgets.interact(angle=(0, 360))
    def f(angle=5):
        plot_cameras(camera, angle=angle)
    '''

    # Get all camera poses
    poses = []
    for camera in cameras:
        poses.append(camera.c2w)
    poses = torch.stack(poses, dim=0).cpu().numpy()
    
    # Get all camera centers
    camera_centers = poses[:, :3, 3]
    
    scale = np.max(np.linalg.norm(camera_centers, axis=1)) * 0.2

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Cartesian axes
    ax.quiver(0, 0, 0, 1, 0, 0, length=scale, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color='g')
    ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color='b')
    ax.text(0, 0, 0, 'w')

    # Draw bounding cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="black")

    for i, T_c_0 in enumerate(poses):
        ax.quiver(T_c_0[0, 3], T_c_0[1, 3], T_c_0[2, 3], T_c_0[0, 0], T_c_0[1, 0], T_c_0[2, 0], length=scale, color='r')
        ax.quiver(T_c_0[0, 3], T_c_0[1, 3], T_c_0[2, 3], T_c_0[0, 1], T_c_0[1, 1], T_c_0[2, 1], length=scale, color='g')
        ax.quiver(T_c_0[0, 3], T_c_0[1, 3], T_c_0[2, 3], T_c_0[0, 2], T_c_0[1, 2], T_c_0[2, 2], length=scale, color='b')
        ax.text(T_c_0[0, 3], T_c_0[1, 3], T_c_0[2, 3], str(i))

    min_lim = np.min(camera_centers)
    max_lim = np.max(camera_centers)
    lim = np.max([abs(min_lim), abs(max_lim)]) + scale
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # axis equal
    ax.set_aspect('equal')
    ax.view_init(30, angle)
    plt.show()