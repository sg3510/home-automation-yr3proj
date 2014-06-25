#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup
import os
# from sphinx.setup_command import BuildDoc

__description__ = """Python package for Gaussian process regression in python 

========================================================

demo_gpr.py explains how to perform basic regression tasks.
demo_gpr_robust.py shows how to apply EP for robust Gaussian process regression.

gpr.py Basic gp regression package
gpr_ep.py GP regression with EP likelihood models

covar: covariance functions"""

pygppath = os.path.join('lib', 'python2.7', 'dist-packages', 'pygp')

def get_recursive_data_files(path):
    out = []
    for (p, d, files) in os.walk(path):
        files = [os.path.join(p, f) for f in files]
        out.append((p, files))
    return out

setup(name='pygp',
      version='1.1.09',
      description=__description__,
      author="Oliver Stegle, Max Zwiessele, Nicolo Fusi",
      author_email='EMAIL HERE',
      url='https://github.com/PMBio/pygp',
      packages=['pygp', 'pygp.covar',
                'pygp.gp', 'pygp.likelihood',
                'pygp.linalg', 'pygp.optimize',
                'pygp.plot', 'pygp.priors',
                'pygp.demo', 'pygp.util',
                # 'pygp.doc'
                ],
      package_dir={'pygp': 'pygp'},
      install_requires=['scipy', 'sphinx'],
      include_package_data=True,
      data_files=get_recursive_data_files('doc'),  # [(pygppath, ['README.txt', 'LICENSE.txt'])] +
      # cmdclass = {'build_sphinx': BuildDoc},
      license='GPLv2',
      requires=['numpy', 'scipy', 'sphinx']
      )
