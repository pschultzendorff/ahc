from setuptools import setup, find_packages

setup(
    name="tpf_lab",
    version="1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
