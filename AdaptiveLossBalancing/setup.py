from setuptools import setup, find_packages

setup(
    # package information
    name="AdaptiveLossBalancing",
    version="1.0.0",
    description=(
        "Adaptive per-loss weighting for multi-loss training: normalise each loss "
        "by its own scale and steer weight toward lagging losses, wrapping any "
        "torch optimizer."
    ),
    author="Jeff Rhoades <rhoadesj@hhmi.org>",
    packages=find_packages(),
    # dependencies
    install_requires=[
        "setuptools",
        "wheel",
        "torch",
        "matplotlib",
    ],
    extras_require={"test": ["pytest"]},
    python_requires=">=3.9",
    # build system
    build_backend="setuptools.build_meta",
)
