from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="noisereduce",
    packages=find_packages(),
    version="3.0.2",
    description="Noise reduction using Spectral Gating in Python",
    author="Tim Sainburg",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timsainb/noisereduce",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["scipy", "matplotlib", "librosa", "numpy", "tqdm"],
    extras_require={
        "PyTorch": ["torch>=1.9.0"],
    },
)
