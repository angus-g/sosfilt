project(
  'sosfilt',
  'c', 'cpp', 'cython',
  version: '1.3.0',
  license: 'MIT',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
pythran = find_program('pythran', native: true)
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])

incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py3,
    [
      '-c',
      'import numpy as np; print(np.get_include())',
    ],
    check: true
  ).stdout().strip()
endif
np_inc = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: np_inc)

incdir_pythran = meson.get_external_property('pythran-include-dir', 'not-given')
if incdir_pythran == 'not-given'
  incdir_pythran = run_command(py3,
    [
      '-c',
      'import pythran; print(pythran.get_include())',
    ],
    check: true
  ).stdout().strip()
endif
pythran_dep = declare_dependency(include_directories: incdir_pythran)

blas = dependency(['openblas', 'OpenBLAS'])
lapack = dependency(['openblas', 'OpenBLAS'])

cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '@INPUT@']
cython_c_args = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
cython_gen = generator(cython,
  arguments: cython_args,
  output: '@BASENAME@.c',
)

pythran_cpp_args = ['-DENABLE_PYTHON_MODULE']

subdir('sosfilt')
