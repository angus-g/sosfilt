from setuptools import Extension, setup

setup(
    ext_modules=[Extension("sosfilt._sosfilt", ["sosfilt/_sosfilt.pyx"])],
)