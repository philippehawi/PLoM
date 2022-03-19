# File: setup.py
# File Created: Friday, 18th March 2022 4:24:10 pm
# Author: Philippe Hawi (hawi@usc.edu)

from setuptools import setup

description = (
    "Probabilistic Learning on Manifolds"
)

dependencies = [
    "numpy",
    "matplotlib",
    "scipy",
    "joblib"
]

scripts = ["scripts/plom_run.py", "scripts/plom_make_input_template.py"]

setup(
    name="plom",
    version="0.6.0",
    description=description,
    url="https://github.com/philippehawi/PLoM",
    author="Philippe Hawi",
    author_email="hawi@usc.edu",
    license="MIT",
    install_requires=dependencies,
    py_modules=["plom"],
    scripts=scripts
)
