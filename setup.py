import re

from setuptools import find_packages, setup

version = ""
with open("sail_model_optimizer/__init__.py") as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

requirements = []
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="sail-model-optimizer",
    version=version,
    install_requires=requirements,
    packages=find_packages(),
    package_data={},
    python_requires="==3.8.10",
    author="Jaap Oosterbroek",
    author_email="jaap@secureailabs.com",
    description="A library to support the sail parameter selection and model optimization process.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://nowhere.not",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: SAIL :: Propritary",
        "Operating System :: OS Independent",
    ],
)
