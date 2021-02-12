cimport numpy as np
cimport cython

ctypedef fused DTYPE_t:
    float
    float complex
    double
    double complex
    long double
    long double complex

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _sosfilt(DTYPE_t [:, :, ::1] sos,
             DTYPE_t [:, ::1] x,
             DTYPE_t [:, :, ::1] zi):
    cdef Py_ssize_t n_signals = x.shape[0]
    cdef Py_ssize_t n_samples = x.shape[1]
    cdef Py_ssize_t n_sections = sos.shape[1]
    cdef Py_ssize_t i, n, s
    cdef DTYPE_t x_new, x_cur
    cdef DTYPE_t[:, ::1] zi_slice

    with nogil:
        for i in range(n_signals):
            zi_slice = zi[i, :, :]

            for n in range(n_samples):
                x_cur = x[i, n]

                for s in range(n_sections):
                    x_new = sos[i, s, 0] * x_cur + zi_slice[s, 0]
                    zi_slice[s, 0] = sos[i, s, 1] * x_cur - sos[i, s, 4] * x_new + zi_slice[s, 1]
                    zi_slice[s, 1] = sos[i, s, 2] * x_cur - sos[i, s, 5] * x_new
                    x_cur = x_new

                x[i, n] = x_cur
