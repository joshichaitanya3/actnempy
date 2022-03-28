import pathlib
import sys

from setuptools import find_packages
from setuptools import setup


assert sys.version_info >= (3, 6, 0), "actnempy requires Python 3.6+"

NAME = "actnempy"
DESCRIPTION = "Analysis Suite for 2D Active Nematics"
URL = "https://github.com/joshichaitanya3/actnempy"
EMAIL = "chaitanyajoshi.usa@gmail.com, matthew.se.peterson@gmail.com, mike.m.norton@gmail.com"
AUTHOR = "Chaitanya Joshi, Matthew S. E. Peterson, Michael M. Norton"
PYTHON = ">=3.6"
LICENSE = "MIT"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Physics",
]

here = pathlib.Path(__file__).parent

with open(here / "requirements.txt", "r") as f:
    REQUIRED = f.readlines()

# with open(here / "README.rst", "r") as f:
#     LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    description=DESCRIPTION, # long_description=LONG_DESCRIPTION, long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=REQUIRED,
    python_requires=PYTHON,
    license=LICENSE,
    classifiers=CLASSIFIERS,
)
