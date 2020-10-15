from setuptools import Extension, setup
import numpy

setup(
    ext_modules=[Extension("sosfilt._sosfilt", ["sosfilt/_sosfilt.pyx"])],
    include_dirs=[numpy.get_include()],
)
