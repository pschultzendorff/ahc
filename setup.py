from setuptools import find_packages, setup

setup(
    name="ahc",
    version="1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
