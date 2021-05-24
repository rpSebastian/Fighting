#!/usr/bin/env python
import datetime
import os

from setuptools import find_packages, setup

readme = open("README.md").read()

# TODO: automate version
VERSION = "0.0.1"
VERSION += ".dev" + datetime.datetime.now().strftime("%Y%m%d%H%M")

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    requirements = [line.strip() for line in f.readlines()]

with open(os.path.join(this_dir, "requirements-dev.txt"), "r") as f:
    requirements_dev = [line.strip() for line in f.readlines()]

extras = {
    "dev": requirements_dev,
}

setup(
    # Metadata
    name="malib",
    version=VERSION,
    author="CASIA-GML",
    author_email="",
    url="",
    description="MALib - Multi Agent Reinforcement Learning Framework",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    # Package info
    packages=find_packages(exclude=("tests*", "docs*", "examples*")),
    zip_safe=True,
    install_requires=requirements,
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
