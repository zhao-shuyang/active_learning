"""
@authors Zhao Shuyang (contact@zhaoshuyang.com)
@date 2021
"""


import os
import active_learning

from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as command


setup(
    name='active_learning',
    version= '0.1.0',
    description='active_learning is an active learning package that contains a few implementations of proximity-based active learning methods.',
    author='Zhao Shuyang',
    author_email='contact@zhaoshuyang.com',
    license='LICENSE',
    install_requires=[
        'matplotlib==3.5.0',
        'numpy==1.21.4',
        'scikit-learn==1.0.1',
        'scipy==1.7.3',
        'pandas',
        'soundfile'        
    ]
    )
