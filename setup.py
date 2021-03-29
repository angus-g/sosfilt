from setuptools import Extension, setup
from pythran.dist import PythranExtension
import numpy

setup(
    ext_modules=[
        Extension("sosfilt._sosfilt", ["sosfilt/_sosfilt.pyx"]),
        PythranExtension("sosfilt._zpk_funcs", ["sosfilt/_zpk_funcs.py"]),
    ],
    include_dirs=[numpy.get_include()],
)
