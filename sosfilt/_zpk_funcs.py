import numpy as np

# pythran export _cplxreal(complex[])
def _cplxreal(z):
    tol = 100 * np.finfo((1.0 * z).dtype).eps

    z = z[np.lexsort((abs(z.imag), z.real))]
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        return np.array([]), zr

    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError("Array contains complex value without conjugate")

    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = np.diff(np.concatenate(([0], same_real, [0])))
    run_starts = np.nonzero(diffs > 0)[0]
    run_stops = np.nonzero(diffs < 0)[0]

    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[:] = chunk[np.lexsort([abs(chunk.imag)])]

    zc = (zp + zn.conj()) / 2
    return zc, zr


def _nearest_real_complex_idx(fro, to, which):
    order = np.argsort(np.abs(fro - to))
    mask = np.isreal(fro[order])
    if which == "complex":
        mask = ~mask
    return order[np.nonzero(mask)[0][0]]


def poly(zeros):
    dt = zeros.dtype
    a = np.ones((1,), dtype=dt)
    for k in range(len(zeros)):
        a = np.convolve(a, np.array([1, -zeros[k]], dtype=dt), mode="full")

    return a


def zpk2tf(z, p, k):
    b = k * poly(z)
    a = poly(p)

    # this is a bit of an assumption to make...
    return b.real, a.real


def zpk2sos(z, p, k, n_sections):
    sos = np.zeros((n_sections, 6))

    if len(p) % 2 == 1:
        p = np.append(p, 0)
        z = np.append(z, 0)

    z = np.concatenate(_cplxreal(z))
    p = np.concatenate(_cplxreal(p))

    p_sos = np.zeros((n_sections, 2), np.complex128)
    z_sos = np.zeros_like(p_sos)
    for si in range(n_sections):
        # select the next "worst" pole
        p1_idx = np.argmin(np.abs(1 - np.abs(p)))
        p1 = p[p1_idx]
        p = np.delete(p, p1_idx)

        # pair that pole with a zero

        if np.isreal(p1) and np.isreal(p).sum() == 0:
            # first order section
            z1_idx = _nearest_real_complex_idx(z, p1, "real")
            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)
            p2 = z2 = 0
        else:
            if not np.isreal(p1) and np.isreal(z).sum() == 1:
                # choose a complex zero to pair with
                z1_idx = _nearest_real_complex_idx(z, p1, "complex")
                assert not np.isreal(z[z1_idx])
            else:
                z1_idx = np.argmin(np.abs(p1 - z))

            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)

            # we have p1 and z1, figure out p2 and z2
            if not np.isreal(p1):
                if not np.isreal(z1):  # complex pole, complex zero
                    p2 = p1.conj()
                    z2 = z1.conj()
                else:  # complex pole, real zero
                    p2 = p1.conj()
                    z2_idx = _nearest_real_complex_idx(z, p1, "real")
                    z2 = z[z2_idx]
                    assert np.isreal(z2)
                    z = np.delete(z, z2_idx)
            else:
                if not np.isreal(z1):  # real pole, complex zero
                    z2 = z1.conj()
                    p2_idx = _nearest_real_complex_idx(p, z1, "real")
                    p2 = p[p2_idx]
                    assert np.isreal(p2)
                else:  # real pole, real zero
                    # next "worst" pole
                    idx = np.nonzero(np.isreal(p))[0]
                    assert len(idx) > 0
                    p2_idx = idx[np.argmin(np.abs(np.abs(p[idx]) - 1))]
                    p2 = p[p2_idx]
                    assert np.isreal(p2)
                    z2_idx = _nearest_real_complex_idx(z, p2, "real")
                    z2 = z[z2_idx]
                    assert np.isreal(z2)
                    z = np.delete(z, z2_idx)
                p = np.delete(p, p2_idx)

        p_sos[si] = [p1, p2]
        z_sos[si] = [z1, z2]

    assert len(p) == len(z) == 0
    del p, z

    p_sos = p_sos[::-1]
    z_sos = z_sos[::-1]
    gains = np.ones(n_sections, np.array(k).dtype)
    gains[0] = k
    for si in range(n_sections):
        x = zpk2tf(z_sos[si], p_sos[si], gains[si])
        sos[si] = np.concatenate(x)

    return sos


# pythran export zpk2sos_multiple(complex[][], complex[][], float[])
def zpk2sos_multiple(z, p, k):
    nfilt = len(k)
    assert z.shape[1] == p.shape[1] == nfilt

    # pad p and z to the same length
    p = np.concatenate((p, np.zeros((max(z.shape[0] - p.shape[0], 0), nfilt))), axis=0)
    z = np.concatenate((z, np.zeros((max(p.shape[0] - z.shape[0], 0), nfilt))), axis=0)

    n_sections = (max(p.shape[0], z.shape[0]) + 1) // 2
    sos = np.zeros((nfilt, n_sections, 6))

    for filt in range(nfilt):
        sos[filt, :, :] = zpk2sos(z[:, filt], p[:, filt], k[filt], n_sections)

    return sos
