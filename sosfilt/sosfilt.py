import numpy as np
from scipy.signal import signaltools
from ._sosfilt import _sosfilt

def _validate_sos(sos):
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    if not (sos[:, 3] == 1).all():
        raise ValueError("sos[:, 3] should be all ones")

    return sos, n_sections

def sosfilt_zi(sos):
    sos = np.asarray(sos)
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("sos must be shape (n_sections, 6)")

    if sos.dtype.kind in "bui":
        sos = sos.astype(np.float64)

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2), dtype=sos.dtype)
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        zi[section] = scale * signaltools.lfilter_zi(b, a)
        scale *= b.sum() / a.sum()

    return zi

def sosfilt(sos, x, axis=-1, zi=None):
    x = signaltools._validate_x(x)

    sos, n_sections = _validate_sos(sos)

    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    inputs = [sos, x]

    if zi is not None:
        inputs.append(np.asarray(zi))

    dtype = np.result_type(*inputs)
    if dtype.char not in "fdgFDGO":
        raise NotImplementedError(f"input type '{dtype}' not supported")

    if zi is not None:
        zi = np.array(zi, dtype)
        if zi.shape != x_zi_shape:
            raise ValueError("invalid zi shape")
        return_zi = True
    else:
        zi = np.zeros(x_zi_shape, dtype=dtype)
        return_zi = False

    axis = axis % x.ndim
    x = np.moveaxis(x, axis, -1)
    zi = np.moveaxis(zi, [0, axis + 1], [-2, -1])
    x_shape, zi_shape = x.shape, zi.shape
    x = np.reshape(x, (-1, x.shape[-1]))
    x = np.array(x, dtype, order="C") # make a modifiable copy
    zi = np.ascontiguousarray(np.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)
    _sosfilt(sos, x, zi)
    x.shape = x_shape
    x = np.moveaxis(x, -1, axis)

    if return_zi:
        zi.shape = zi_shape
        zi = np.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x

    return out

def sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None):
    # filter validation
    # XXX: needs to be modified to handle mutliple filters
    sos, n_sections = _validate_sos(sos)

    # input validation
    x = signaltools._validate_x(x)

    # padding validation
    # XXX: validate ntaps is consistent across filters
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    edge, ext = signaltools._validate_pad(padtype, padlen, x, axis, ntaps=ntaps)

    # filter initial conditions
    # XXX: handle multiple filters
    zi = sosfilt_zi(sos)
    zi_shape = [1] * x.ndim
    zi_shape[axis] = 2
    zi.shape = [n_sections] + zi_shape

    # forward filter
    x_0 = signaltools.axis_slice(ext, stop=1, axis=axis)
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x_0)

    # backward filter
    y_0 = signaltools.axis_slice(y, start=-1, axis=axis)
    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y_0)

    # reshaping
    y = signaltools.axis_reverse(y, axis=axis)
    if edge > 0:
        y = signaltools.axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y