from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from Cython.Distutils import build_ext
from numpy import get_include as numpy_get_include

ext_modules = [Extension("_regions", ["src/_regions.pyx"], 
                         include_dirs = [numpy_get_include()])]

setup(
    name = '_regions',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)