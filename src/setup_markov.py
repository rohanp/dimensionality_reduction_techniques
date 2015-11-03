from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os

# usage: python setup.py build_ext --inplace
# you will have to change include_dirs, library_dirs, and extra_compile_args
# to the absolute path of the Include directory in your system
os.environ["CXX"] = "gcc"

setup(
    author = "Rohan Pandit",
    ext_modules = cythonize([Extension("calcMarkov", 
    					sources = ["calcMarkov.pyx"],
    					)])
   							
   	)

