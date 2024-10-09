from setuptools import setup

setup(
    # package information
    name="KnowledgeExpansion",
    version="1.0.0",
    description="Experiments beyond knowledge distillation",
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
    ],
    # build system
    build_backend="setuptools.build_meta",
)
