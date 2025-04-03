#!/usr/bin/env python

from setuptools import setup, find_packages
import os

setup(name='frostie',
      version='0.9.0',
      description='An open source modelling and retrieval package for reflectance spectroscopy of planetary surfaces',
      author='Ishan Mishra',
      author_email='ishan.mishra@jpl.nasa.gov',
      packages=['frostie'],
      include_package_data=True,
      license = 'MIT License',
      install_requires=[
          'numpy',
          'matplotlib',
          'jupyter',
          'dynesty',
      ],
      zip_safe=False)

