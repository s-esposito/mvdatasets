from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


scene_path = Path("/home/stefano/Data/kubric/static")

# get all segmentation paths
segs_dir = scene_path / "segmentation"
segs_paths = sorted(list(segs_dir.glob("*.png")))

all_unique_segs = set()
all_segs = []
pbar_segs = tqdm(segs_paths, desc="segs", ncols=100)
for seg_path in pbar_segs:
    seg = np.array(Image.open(seg_path))
    # get unique values
    unique_segs = np.unique(seg)
    all_unique_segs.update(unique_segs)
    all_segs.append(seg)

# print unique values
print("all_unique_segs", all_unique_segs)

# # vis first segmentation
# seg = all_segs[0]  # [..., None]
# print("seg.shape", seg.shape)
# print("seg.dtype", seg.dtype)
# print("seg.unique", np.unique(seg))
# n = np.max(np.unique(seg))

# from_list = matplotlib.colors.LinearSegmentedColormap.from_list
# cm = from_list(None, plt.cm.Set1(range(0,n)), n)
# plt.imshow(seg, cmap=cm)
# plt.clim(-0.5, n-0.5)
# cb = plt.colorbar(ticks=range(0, n), label='Group')
# # cb.ax.tick_params(lenght=0)
# plt.show()

# all_masks = []
# for seg in all_segs:
#     mask = np.zeros_like(seg)
#     mask[seg == 1] = 1
#     all_masks.append(mask)
    
# # vis first mask
# mask = all_masks[0]
# plt.imshow(mask, cmap="gray")
# plt.show()