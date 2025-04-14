"""Setup for the splinecalib package."""

import setuptools
import numpy

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    ext_modules=[setuptools.Extension("loss_fun_c", ["splinecalib/loss_fun_c.c"],
                                      include_dirs=[numpy.get_include()])],
)
