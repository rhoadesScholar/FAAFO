from setuptools import setup, find_packages

setup(
    # package information
    name="ProbabilisticOptimizers",
    version="1.0.0",
    description=(
        "A family of optimizers that wrap any off-the-shelf torch optimizer and "
        "probabilistically resample high-gradient parameters, per layer, via a "
        "softmax over gradient magnitudes."
    ),
    author="Jeff Rhoades <rhoadesj@hhmi.org>",
    packages=find_packages(),
    # dependencies
    install_requires=[
        "setuptools",
        "wheel",
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
    ],
    extras_require={"test": ["pytest"]},
    python_requires=">=3.9",
    # build system
    build_backend="setuptools.build_meta",
)
