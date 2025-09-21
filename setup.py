"""
Setup script for Stem Cell Therapy Analysis package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stem-cell-therapy-analysis",
    version="1.0.0",
    author="Benjamin Hu",
    author_email="your.email@example.com",
    description="Advanced statistical & AI framework for stem cell therapy clinical trial analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/stem-cell-therapy-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stem-cell-analysis=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "stem-cell-therapy",
        "clinical-trials",
        "machine-learning",
        "statistical-analysis",
        "medical-ai",
        "healthcare-analytics",
        "biostatistics",
        "data-science"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/stem-cell-therapy-analysis/issues",
        "Source": "https://github.com/your-username/stem-cell-therapy-analysis",
        "Documentation": "https://stem-cell-therapy-analysis.readthedocs.io/",
    },
)