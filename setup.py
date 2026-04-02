from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cofgen",
    version="0.1.0",
    description="COF property prediction, PXRD simulation, and generative design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="COFGen Authors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "networkx>=3.0",
        "matplotlib>=3.7",
        "pandas>=2.0",
    ],
    extras_require={
        "train": [
            "torch>=2.0",
            "torch-geometric>=2.3",
            "pymatgen>=2023.1",
        ],
        "rdkit": ["rdkit-pypi>=2023.3"],
        "mattersim": ["mattersim>=1.0"],
        "pycofbuilder": ["pyCOFBuilder>=0.0.14"],
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "cofgen=cofgen_tool:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="COF covalent-organic-framework PXRD property-prediction materials-science",
)
