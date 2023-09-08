import numpy as np
import cv2 as cv

class Camera():
    
    def __init__(self, c2w, K):
        self.c2w = c2w  # torch.tensor of shape (3, 4) or (4, 4)
        self.K = K      # torch.tensor of shape (3, 3)
        

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def K_Rt_from_P(filename, P=None):
    
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose