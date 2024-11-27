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
        "torch>=2.3.1",
        "pillow>=9.4.0",
        "torchvision>=0.18.1",
        "iopath>=0.1.10",
        "matplotlib>=3.9.1",
        "jupyter>=1.0.0",
        "opencv-python>=4.7.0",
        "open3d>=0.18.0",
        "rich>=13.9.4",
        "tyro==0.9.2",
        "ffmpeg==1.4.0",
        "pycolmap@git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e",
    ],
)
