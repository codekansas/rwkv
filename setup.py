#!/usr/bin/env python
"""Setup script for the RWKV project."""

import re

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("rwkv/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in rwkv/__init__.py"
version: str = version_re.group(1)


setup(
    name="rwkv-ml",
    version=version,
    description="ML project template repository",
    author="Benjamin Bolte",
    url="https://github.com/codekansas/rwkv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    install_requires=["ml-starter"],
    tests_require=["ml-starter[dev]"],
    extras_require={"dev": ["ml-starter[dev]"]},
    package_data={"rwkv": ["py.typed"]},
)
