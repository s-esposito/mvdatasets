from setuptools import setup, find_packages

setup(
    name="mvdatasets",
    version="1.0.0",
    description="Standardized DataLoaders for 3D Computer Vision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/s-esposito/mvdatasets",
    author="Stefano Esposito",
    author_email="stefano.esposito@uni-tuebingen.de",
    packages=find_packages(),
    install_requires=[
        # docs
        "sphinx>=8.1.3",
        "sphinx-rtd-theme>=3.0.2",
        "sphinxcontrib-mermaid>=1.0.0",
        "pip install sphinxcontrib-bibtex>=2.6.3",
        "myst-parser>=4.0.0",
        # main
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
    tests_require=["pytest"],
    python_requires=">=3.6",
)
