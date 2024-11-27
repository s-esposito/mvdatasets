from mvdatasets import MVDataset
from mvdatasets.utils.printing import print_error

if __name__ == "__main__":
    # Example to trigger an exception
    class PointCloud:
        pass

    def subtract(a, b):
        return a - b

    pc = PointCloud()
    subtract(5.0, pc)  # This will raise a TypeError
    
    print_error("test")