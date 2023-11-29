class Camera:
    def __init__(self, val):
        self.val = val
        self.multiplier = 1
    
    def __str__(self):
        return "val " + str(self.val) + " multiplier " + str(self.multiplier)

cameras_train = []
for i in range(10):
    cameras_train.append(Camera(i))

cameras_test = []
for i in range(10):
    cameras_test.append(Camera(i))

cameras_splits = {}
cameras_splits["train"] = cameras_train
cameras_splits["test"] = cameras_test

for key, cameras_list in cameras_splits.items():
    print(key)
    for camera in cameras_list:
        print(camera)

# # check if change applied to dict applies to list
# print("changing multiplier val in dict")
# for key, cameras_list in cameras_splits.items():
#     for camera in cameras_list:
#         camera.multiplier = 2

# print("checking multiplier val in list")
# for camera in cameras_train:
#     print(camera)

print("changing multiplier val in list")
cameras_all = []
for key, cameras_list in cameras_splits.items():
    cameras_all += cameras_list
    
for camera in cameras_all:
    camera.multiplier = 2

print("checking multiplier val in dict")
for key, cameras_list in cameras_splits.items():
    print(key)
    for camera in cameras_list:
        print(camera)