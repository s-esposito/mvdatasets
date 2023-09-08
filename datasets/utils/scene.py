import os
import sys

class Scene():
    
    def __init__(self, dataset_name, scene_name, data_path):
        
        self.dataset = dataset_name
        self.name = scene_name
        self.path = data_path
        
        # check if path exists
        if not os.path.exists(data_path):
            print("ERROR: path does not exist")
            sys.exit()