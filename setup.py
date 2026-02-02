"""Setup configuration for PhIO (Physics-Informed Optimizer)."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="phio",
    version="0.1.0",
    author="PhIO Contributors",
    author_email="phio-dev@example.com",
    description="Physics-Informed Optimizer: Production-grade PINNs for solving PDEs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sinsangwoo/physics-informed-optimizer",
    packages=find_packages(exclude=["tests", "examples", "benchmarks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "phio=phio.cli:main",
        ],
    },
)