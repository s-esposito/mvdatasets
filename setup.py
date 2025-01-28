from setuptools import setup, find_packages

setup(
    name="mvdatasets",
    version="0.6",
    description="common multi-view datasets loaders",
    url="https://github.com/s-esposito/mv_datasets",
    author="Stefano Esposito",
    author_email="stefano.esposito@uni-tuebingen.de",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.4",
        "tqdm>=4.66.1",
        "torch>=2.1.0",
        "pillow>=9.4.0",
        "torchvision>=0.16.0",
        "iopath>=0.1.10",
        "matplotlib",
        "jupyter>=1.0.0",
        "opencv-python>=4.7.0",
        "open3d>=0.18.0",
        "rich>=13.9.4",
        "pycolmap==0.4.0"
    ],
    python_requires=">=3.8",
)
