import numpy as np
from scipy.signal import signaltools
from ._sosfilt import _sosfilt


def _validate_sos(sos):
    sos = np.atleast_2d(sos)

    if sos.ndim == 2:
        return sos, -1, 1

    n_filters, n_sections, m = sos.shape

    if m != 6:
        raise ValueError("last dim of sos array must be size 6")
    if not (sos[..., 3] == 1).all():
        raise ValueError("sos[..., 3] should be all ones")

    return sos, n_sections, n_filters


def _validate_nfilters(x, axis, n_filters):
    # with 1 filter, we'll broadcast it as usual
    if n_filters == 1:
        return

    # otherwise, the number of filters must match the
    # product of the non-filtered axes of x
    x_shape = list(x.shape)
    x_shape.pop(axis)
    if n_filters != np.product(x_shape):
        raise ValueError("n_filters must match product of non-filtered axes")


def _validate_ntaps(ntaps):
    first_tap = ntaps[0]
    if not np.all(ntaps == first_tap):
        raise ValueError("all filters must have the same number of taps")

    return first_tap


def sosfilt_zi(sos):
    sos = np.asarray(sos)
    if sos.ndim != 3 or sos.shape[2] != 6:
        raise ValueError("sos must be shape (n_filters, n_sections, 6)")

    if sos.dtype.kind in "bui":
        sos = sos.astype(np.float64)

    n_filters, n_sections = sos.shape[:2]
    zi = np.empty((n_filters, n_sections, 2), dtype=sos.dtype)
    scale = np.ones(n_filters, dtype=sos.dtype)
    for section in range(n_sections):
        b = sos[:, section, :3]
        a = sos[:, section, 3:]

        # lfilter_zi solves zi = A*zi + B
        # where A = scipy.linalg.companion(a).T
        # and   B = b[1:] - a[1:]*b[0]
        #
        # because a[0] = 1 for a sos filter, we have
        # A = [ 1 + a[1], -1;
        #         a[2],    1]
        # so zi[0] = (B[0] + B[1]) / (1 + a[1] + a[2])
        # and zi[1] = B[1] - a[2] * zi[0]
        #
        # we can pretty easily write this in a vectorised
        # way over all n_filters!

        B = b[:, 1:] - a[:, 1:] * b[:, [0]]
        zi[:, section, 0] = B.sum(axis=-1) / a.sum(axis=-1)
        zi[:, section, 1] = B[:, 1] - a[:, 2] * zi[:, section, 0]
        zi[:, section, :] *= scale[:, None]

        scale *= b.sum(axis=-1) / a.sum(axis=-1)

    return zi


def sosfilt(sos, x, axis=-1, zi=None):
    x = signaltools._validate_x(x)

    sos, n_sections, n_filters = _validate_sos(sos)

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
    x = np.array(x, dtype, order="C")  # make a modifiable copy
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
    """
    A forward-backward digital filter using cascaded second-order sections.

    Parameters
    ----------
    sos : array_like
        An array of second-order filter coefficients. It must have either
        the shape ``(n_filters, n_sections, 6)`` or ``(n_sections, 6)``.
        In the latter case, the single filter will be broadcast over the
        whole input array. In the former case, `n_filters` must match
        the product of the non-filter axes of `x`. Additionally,
        `n_sections` must be the same for all filters.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None. Determines the type
        of extension to use for the padded signal to which the filter is
        applied. If None, no padding is used. The default is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.
    """

    # filter validation
    sos, n_sections, n_filters = _validate_sos(sos)

    if n_filters == 1:
        # defer to scipy's implementation for the usual case of a single filter
        return signaltools.sosfiltfilt(
            sos, x, axis=axis, padtype=padtype, padlen=padlen
        )

    # input validation
    x = signaltools._validate_x(x)
    _validate_nfilters(x, axis, n_filters)

    # padding validation
    ntaps = np.ones(n_filters, dtype=int) * (2 * n_sections + 1)
    ntaps -= np.minimum((sos[..., 2] == 0).sum(axis=1), (sos[..., 5] == 0).sum(axis=1))
    ntaps = _validate_ntaps(ntaps)
    edge, ext = signaltools._validate_pad(padtype, padlen, x, axis, ntaps=ntaps)

    # filter initial conditions
    zi = sosfilt_zi(sos)

    # to handle multiple filters, we might want zi.shape = (n_sections, ..., 2, ...)
    zi = np.swapaxes(zi, 0, 1)  # => (n_sections, n_filters, 2)
    zi_shape = list(x.shape)
    zi_shape[axis] = 2
    # we need to swap axis to the end first, so it picks up the end of zi
    zi_shape[axis], zi_shape[-1] = zi_shape[-1], zi_shape[axis]
    zi_shape = [n_sections] + zi_shape
    zi = zi.reshape(zi_shape)  # should look like (n_sections, ..., 2)
    # now we need to swap axis back from the end to where it should be
    zi = np.swapaxes(zi, -1, (axis % x.ndim) + 1)

    # forward filter
    x_0 = signaltools.axis_slice(ext, stop=1, axis=axis)
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x_0)

    # backward filter
    y_0 = signaltools.axis_slice(y, start=-1, axis=axis)
    (y, zf) = sosfilt(
        sos, signaltools.axis_reverse(y, axis=axis), axis=axis, zi=zi * y_0
    )

    # reshaping
    y = signaltools.axis_reverse(y, axis=axis)
    if edge > 0:
        y = signaltools.axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y
