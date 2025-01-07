from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("spectral_accel.pyx"),
    include_dirs=[np.get_include()]
)

# To compile the Cython file and generate the extension module, run the following command:
# 
# python setup.py build_ext --inplace
#
# Working directory:
# ./audas_process/modules/spectral_processor
#
# This command compiles the Cython file and generates the extension module in the current directory.
