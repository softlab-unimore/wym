import os
import re

from setuptools import setup, find_packages


def read(*names, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *names)) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt') as f:
    required = f.read().splitlines()

VERSION = find_version('wym', '__init__.py')
long_description = read('README.rst')

setup(
    name='wym',
    description='An intrisecally explainable model for Entity Matching',
    long_description=long_description,
    version=VERSION,
    author='Andrea Baraldi',
    author_email='baraldian@gmail.com',
    url='https://github.com/softlab-unimore/wym',
    license='MIT',
    packages=find_packages(exclude=('*test*','*run_experiments*')),
    install_requires=required,
    python_requires='>=3.7'
)

