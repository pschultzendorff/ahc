from setuptools import find_packages, setup

setup(
    name="tpf",
    version="1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
