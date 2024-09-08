from setuptools import setup, find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))

VERSION = '0.1'
DESCRIPTION = 'Utilities for efficiently iterating over mini-batches of PyTorch tensors'
LONG_DESCRIPTION = ('This package allows eliminating the over-head incurred by the DataLoader class when iterating '
                    'over in-memory tensors for training small models. Allows iterating over (shuffled) samples, '
                    'or groups of samples, such as what is required for learning-to-rank.')

setup(
    name='batch_iter',
    version=VERSION,
    author='Alex Shtoff',
    author_email='<alex.shtf@gmail.com>',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
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
