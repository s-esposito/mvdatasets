from setuptools import setup, find_packages

setup(
    name="mvdatasets",
    version="1.0.0",
    description="Standardized DataLoaders for 3D Computer Vision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/autonomousvision/mvdatasets",
    author="Stefano Esposito",
    author_email="stefano.esposito@uni-tuebingen.de",
    packages=find_packages(),
    install_requires=[
        # main
        "pyquat>=0.5.1",
        "PyOpenGL>=3.1.9",
        "glfw>=2.8.0",
        "gdown>=5.2.0",
        "flake8>=7.1.1",
        "black>=24.8.0",
        "ffmpeg>=1.4.0",
        "pyyaml>=6.0",
        "numpy>=1.21.6",
        "tqdm>=4.67.1",
        "torch>=1.13.1",
        "pillow>=9.5.0",
        "torchvision>=0.14.1",
        "iopath>=0.1.10",
        "matplotlib>=3.5.3",
        "jupyter>=1.1.1",
        "opencv-python>=4.11.0.86",
        "open3d>=0.18.0",
        "rich>=13.8.1",
        "tyro>=0.9.11",
        "pycolmap@git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e",
    ],
    extras_require={
        "tests": [
            # tests
            "pytest"
        ],
        "docs": [
            # docs
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-mermaid",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-osexample",
            "myst-parser",
            "pre-commit",
        ],
    },
    python_requires=">=3.8",
)
