# FROM: gsplat
# undistortion
# self.mapx_dict = dict()
# self.mapy_dict = dict()
# self.roi_undist_dict = dict()
# for camera_id in self.params_dict.keys():
#     params = self.params_dict[camera_id]
#     if len(params) == 0:
#         continue  # no distortion
#     assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
#     assert (
#         camera_id in self.params_dict
#     ), f"Missing params for camera {camera_id}"
#     K = self.Ks_dict[camera_id]
#     width, height = self.imsize_dict[camera_id]

#     if camtype == "perspective":
#         K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
#             K, params, (width, height), 0
#         )
#         mapx, mapy = cv2.initUndistortRectifyMap(
#             K, params, None, K_undist, (width, height), cv2.CV_32FC1
#         )
#         mask = None
#     elif camtype == "fisheye":
#         fx = K[0, 0]
#         fy = K[1, 1]
#         cx = K[0, 2]
#         cy = K[1, 2]
#         grid_x, grid_y = np.meshgrid(
#             np.arange(width, dtype=np.float32),
#             np.arange(height, dtype=np.float32),
#             indexing="xy",
#         )
#         x1 = (grid_x - cx) / fx
#         y1 = (grid_y - cy) / fy
#         theta = np.sqrt(x1**2 + y1**2)
#         r = (
#             1.0
#             + params[0] * theta**2
#             + params[1] * theta**4
#             + params[2] * theta**6
#             + params[3] * theta**8
#         )
#         mapx = fx * x1 * r + width // 2
#         mapy = fy * y1 * r + height // 2

#         # Use mask to define ROI
#         mask = np.logical_and(
#             np.logical_and(mapx > 0, mapy > 0),
#             np.logical_and(mapx < width - 1, mapy < height - 1),
#         )
#         y_indices, x_indices = np.nonzero(mask)
#         y_min, y_max = y_indices.min(), y_indices.max() + 1
#         x_min, x_max = x_indices.min(), x_indices.max() + 1
#         mask = mask[y_min:y_max, x_min:x_max]
#         K_undist = K.copy()
#         K_undist[0, 2] -= x_min
#         K_undist[1, 2] -= y_min
#         roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
#     else:
#         assert_never(camtype)

#     self.mapx_dict[camera_id] = mapx
#     self.mapy_dict[camera_id] = mapy
#     self.Ks_dict[camera_id] = K_undist
#     self.roi_undist_dict[camera_id] = roi_undist
#     self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
#     self.mask_dict[camera_id] = mask