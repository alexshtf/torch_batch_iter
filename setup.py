from setuptools import setup, find_packages
import os


VERSION = '0.1.1'
DESCRIPTION = 'Utilities for efficiently iterating over mini-batches of PyTorch tensors'

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='batch_iter',
    version=VERSION,
    author='Alex Shtoff',
    author_email='<alex.shtf@gmail.com>',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['torch>=2.1.0'],
    keywords=['python', 'pytorch'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3'
    ]
)
