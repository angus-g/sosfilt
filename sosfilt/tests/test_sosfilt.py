import pytest

import numpy as np
from scipy import signal

from sosfilt import sosfiltfilt


class TestFiltFilt(object):
    def filtfilt(self, zpk, x, axis=-1, padtype="odd", padlen=None):
        sos = signal.zpk2sos(*zpk)
        return sosfiltfilt(sos, x, axis, padtype, padlen)

    def test_basic(self):
        zpk = signal.tf2zpk([1, 2, 3], [1, 2, 3])
        out = self.filtfilt(zpk, np.arange(12))
        np.testing.assert_allclose(out, np.arange(12), atol=5.28e-11)

    def test_sine(self):
        rate = 2000
        t = np.linspace(0, 1.0, rate + 1)
        xlow = np.sin(5 * 2 * np.pi * t)
        xhigh = np.sin(250 * 2 * np.pi * t)
        x = xlow + xhigh

        zpk = signal.butter(8, 0.125, output="zpk")
        # magnitude of the largest pole
        r = np.abs(zpk[1]).max()
        eps = 1e-5
        # estimate of number of steps to decay
        # transient by factor of eps
        n = int(np.ceil(np.log(eps) / np.log(r)))

        y = self.filtfilt(zpk, x, padlen=n)
        err = np.abs(y - xlow).max()
        np.testing.assert_(err < 1e-4)

        x2d = np.vstack([xlow, xlow + xhigh])
        y2d = self.filtfilt(zpk, x2d, padlen=n, axis=1)
        np.testing.assert_equal(y2d.shape, x2d.shape)
        err = np.abs(y2d - xlow).max()
        np.testing.assert_(err < 1e-4)

        # test axis keyword
        y2dt = self.filtfilt(zpk, x2d.T, padlen=n, axis=0)
        np.testing.assert_equal(y2d, y2dt.T)

    def test_axis(self):
        x = np.arange(10.0 * 11.0 * 12.0).reshape(10, 11, 12)
        zpk = signal.butter(3, 0.125, output="zpk")
        y0 = self.filtfilt(zpk, x, padlen=0, axis=0)
        y1 = self.filtfilt(zpk, np.swapaxes(x, 0, 1), padlen=0, axis=1)
        np.testing.assert_array_equal(y0, np.swapaxes(y1, 0, 1))
        y2 = self.filtfilt(zpk, np.swapaxes(x, 0, 2), padlen=0, axis=2)
        np.testing.assert_array_equal(y0, np.swapaxes(y2, 0, 2))

    def test_equivalence(self):
        x = np.random.RandomState(0).randn(1000)
        for order in range(1, 6):
            zpk = signal.butter(order, 0.35, output="zpk")
            b, a = signal.zpk2tf(*zpk)
            sos = signal.zpk2sos(*zpk)
            y = signal.filtfilt(b, a, x)
            y_sos = sosfiltfilt(sos, x)
            np.testing.assert_allclose(y, y_sos, atol=1e-12, err_msg=f"order={order}")


def test_multiple():
    x = np.random.RandomState(0).randn(2, 20, 2)
    a, b = x.shape[0], x.shape[2]

    filters = np.stack(
        [
            signal.butter(3, np.random.RandomState(0).rand(), output="sos")
            for _ in range(a * b)
        ]
    )

    y_sos = sosfiltfilt(filters, x, axis=1)

    y = np.empty(x.shape, dtype=x.dtype)

    for i in range(a):
        for j in range(b):
            y[i, :, j] = signal.sosfiltfilt(filters[i * b + j, ...], x[i, :, j])

    np.testing.assert_allclose(y, y_sos)
