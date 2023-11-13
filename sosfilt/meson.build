python_sources = [
  '__init__.py',
  'filter_design.py',
  'sosfilt.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sosfilt',
)

# cython
sosfilt = py3.extension_module('_sosfilt',
  [cython_gen.process('_sosfilt.pyx')],
  dependencies: [np_dep],
  c_args: cython_c_args,
  install: true,
  subdir: 'sosfilt',
)

# pythran
zpk_funcs = custom_target('_zpk_funcs',
  output: ['_zpk_funcs.cpp'],
  input: '_zpk_funcs.py',
  command: [pythran, '-E', '@INPUT@', '-o', '@OUTDIR@/_zpk_funcs.cpp']
)

zpk_funcs = py3.extension_module('_zpk_funcs',
  [zpk_funcs],
  cpp_args: pythran_cpp_args,
  dependencies: [blas, pythran_dep, np_dep],
  install: true,
  subdir: 'sosfilt',
)
