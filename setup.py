"""Setup for the splinecalib package."""

import setuptools
import numpy

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Brian Lucena",
    author_email="brianlucena@gmail.com",
    name='splinecalib',
    license="MIT",
    license_files=['LICENSE'],
    description="SplineCalib is a Python package for calibrating ML models using smoothing splines.  See documentation at: https://splinecalib.readthedocs.io/",
    version='0.0.7',
    long_description=README,
    url='https://github.com/numeristical/splinecalib',
    packages=['splinecalib'],
    package_dir={'splinecalib':
                 'splinecalib'},
    python_requires=">=3.5",
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.3"],
    ext_modules=[setuptools.Extension("loss_fun_c", ["splinecalib/loss_fun_c.c"],
                                      include_dirs=[numpy.get_include()])],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
