"""Setup configuration for PhIO (Physics-Informed Optimizer)."""

from setuptools import find_packages, setup


# Read requirements, filtering out comments and -r directives
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#") and not line.startswith("-r")
        ]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = read_requirements("requirements.txt")
dev_requirements = read_requirements("requirements-dev.txt")

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
        "gpu": ["jax[cuda12_pip]>=0.4.20"],
        "all": dev_requirements + ["jax[cuda12_pip]>=0.4.20"],
    },
    entry_points={
        "console_scripts": [
            "phio=phio.cli:main",
        ],
    },
)
