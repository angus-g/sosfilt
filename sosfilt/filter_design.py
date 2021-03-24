import numpy as np
from scipy import signal

from ._zpk_funcs import zpk2sos_multiple


def _relative_degree(z, p):
    if z.ndim == 1 and p.ndim == 1:
        return len(p) - len(z)
    elif z.ndim == 2 and p.ndim == 2:
        return p.shape[0] - z.shape[0]

    raise ValueError("relative degree between singular and multiple filter")


def lp2lp_zpk(z, p, k, wo=1.0):
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = np.atleast_1d(wo)

    degree = _relative_degree(z, p)

    z_lp = wo * z[:, None]
    p_lp = wo * p[:, None]
    k_lp = k * wo ** degree

    return z_lp, p_lp, k_lp


def lp2hp_zpk(z, p, k, wo=1.0):
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = np.atleast_1d(wo)

    nfilt = len(wo)

    degree = _relative_degree(z, p)

    z_hp = wo / z[:, None]
    p_hp = wo / p[:, None]

    z_hp = np.append(z_hp, np.zeros((degree, nfilt)), axis=0)

    # gain change is applied to the original z, p args, which
    # are only 1d -- so we can just repeat the result
    k_hp = k * np.real(np.prod(-z) / np.prod(-p))
    k_hp = np.repeat(k_hp, nfilt)

    return z_hp, p_hp, k_hp


def bilinear_zpk_multiple(z, p, k, fs):
    z = np.atleast_2d(z)
    p = np.atleast_2d(p)
    k = np.atleast_1d(k)

    nfilt = len(k)

    degree = _relative_degree(z, p)
    fs2 = 2.0 * fs

    # bilinear transform poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # zeros at infinity are moved to Nyquist frequency
    z_z = np.append(z_z, -np.ones((degree, nfilt)), axis=0)

    # compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z, axis=0) / np.prod(fs2 - p, axis=0))

    return z_z, p_z, k_z


def zpk2tf_multiple(z, p, k):
    raise NotImplementedError("zpk2tf not implemented yet")


def butter(N, Wn, btype="lowpass", output="sos", fs=None):
    """
    A modified filter design that will return a suite of
    Butterworth filters, for all the provided Wn values.
    """

    try:
        btype = signal.filter_design.band_dict[btype]
    except KeyError as e:
        raise ValueError("'{}' is an invalid bandtype.".format(btype)) from e

    Wn = np.asarray(Wn)
    if fs is not None:
        Wn = 2 * Wn / fs

    # generate the single Nth-order filter at 1 rad/s cutoff
    z, p, k = signal.buttap(N)

    fs = 2.0
    warped = 2 * fs * np.tan(np.pi * Wn / fs)

    if btype == "lowpass":
        z, p, k = lp2lp_zpk(z, p, k, wo=warped)
    elif btype == "highpass":
        z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    else:
        raise NotImplementedError("'{}' is not a supported bandtype".format(btype))

    # after this point, z, p, k are arrays of multiple filters:
    # for z, p: axis 0 is the "normal" shape; axis 1 is filter number
    # k is an array with length number of filters

    z, p, k = bilinear_zpk_multiple(z, p, k, fs=fs)

    if output == "zpk":
        return z, p, k
    elif output == "ba":
        return zpk2tf_multiple(z, p, k)
    elif output == "sos":
        return zpk2sos_multiple(z.astype(np.complex_), p.astype(np.complex_), k)
