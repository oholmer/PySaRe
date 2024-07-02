import setuptools


description = "PySaRe -- Neural Network-Based Survival Analysis and Reliability Engineering"

with open("README.md") as f:
    long_description, long_description_content_type = f.read(), "text/markdown"

setuptools.setup(
    name="pysare",
    version="0.1.2",
    author="Olov Holmer",
    author_email="olov.holmer@liu.se",
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url="https://github.com/oholmer/PySaRe",
    project_urls={
        "Bug Tracker": "https://github.com/oholmer/PySaRe/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=["numpy>=1.14.0", "torch>=0.4.1","pandas>=2.1","matplotlib>=3.0","torch_lr_finder >=0.2.1","scipy>=1.7.0","lifelines>=0.29.0"],
)