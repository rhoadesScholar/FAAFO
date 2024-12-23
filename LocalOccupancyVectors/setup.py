from setuptools import setup

setup(
    # package information
    name="Local_Occupancy_Vectors",
    version="1.0.0",
    description="Local occupancy vectors for semantic segmentation",
    author="Jeff Rhoades <rhoadesj@hhmi.org>",
    # dependencies
    install_requires=[
        "setuptools",
        "wheel",
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "tensorboard",
        "tensorboardX",
        "matplotlib",
        "monai",
        "ipykernel",
        "pandas",
        "tifffile",
        "einops",
        "cellmap-data@git+https://github.com/janelia-cellmap/cellmap-data",
    ],
    # build system
    build_backend="setuptools.build_meta",
)
