#!/usr/bin/env python3

from setuptools import setup

setup(name='Collision',
      version='0.1',
      description='OpenCL-based collision detection',
      author='Kai Wohlfahrt',
      url='https://github.com/kwohlfahrt/collision',
      packages=['collision'],
      install_requires=['pyopencl'],
      include_package_data=True,
)
