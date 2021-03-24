import pytest

import numpy as np

from sosfilt import _zpk_funcs


class TestCplxReal:
    def test_trivial_input(self):
        np.testing.assert_equal(
            _zpk_funcs._cplxreal(np.array([], dtype=np.complex_)), ([], [])
        )
        np.testing.assert_equal(
            _zpk_funcs._cplxreal(np.array([1], dtype=np.complex_)), ([], [1])
        )

    def test_output_order(self):
        zc, zr = _zpk_funcs._cplxreal(np.roots(np.array([1, 0, 0, 1])))
        np.testing.assert_allclose(
            np.append(zc, zr), [1 / 2 + 1j * np.sin(np.pi / 3), -1]
        )

        eps = np.spacing(1)

        a = [
            0 + 1j,
            0 - 1j,
            eps + 1j,
            eps - 1j,
            -eps + 1j,
            -eps - 1j,
            1,
            4,
            2,
            3,
            0,
            0,
            2 + 3j,
            2 - 3j,
            1 - eps + 1j,
            1 + 2j,
            1 - 2j,
            1 + eps - 1j,  # sorts out of order
            3 + 1j,
            3 + 1j,
            3 + 1j,
            3 - 1j,
            3 - 1j,
            3 - 1j,
            2 - 3j,
            2 + 3j,
        ]
        zc, zr = _zpk_funcs._cplxreal(np.array(a))
        np.testing.assert_allclose(
            zc, [1j, 1j, 1j, 1 + 1j, 1 + 2j, 2 + 3j, 2 + 3j, 3 + 1j, 3 + 1j, 3 + 1j]
        )
        np.testing.assert_allclose(zr, [0, 0, 1, 2, 3, 4])

        z = np.array(
            [
                1 - eps + 1j,
                1 + 2j,
                1 - 2j,
                1 + eps - 1j,
                1 + eps + 3j,
                1 - 2 * eps - 3j,
                0 + 1j,
                0 - 1j,
                2 + 4j,
                2 - 4j,
                2 + 3j,
                2 - 3j,
                3 + 7j,
                3 - 7j,
                4 - eps + 1j,
                4 + eps - 2j,
                4 - 1j,
                4 - eps + 2j,
            ]
        )

        zc, zr = _zpk_funcs._cplxreal(z)
        np.testing.assert_allclose(
            zc, [1j, 1 + 1j, 1 + 2j, 1 + 3j, 2 + 3j, 2 + 4j, 3 + 7j, 4 + 1j, 4 + 2j]
        )
        np.testing.assert_equal(zr, [])

    def test_unmatched_conjugates(self):
        with pytest.raises(ValueError):
            _zpk_funcs._cplxreal(np.array([1 + 3j, 1 - 3j, 1 + 2j]))

        with pytest.raises(ValueError):
            _zpk_funcs._cplxreal(np.array([1 + 3j]))

        with pytest.raises(ValueError):
            _zpk_funcs._cplxreal(np.array([1 - 3j]))


class TestZpk2Sos:
    def sos2zpk(self, z, p, k):
        return _zpk_funcs.zpk2sos_multiple(
            z[:, None], p[:, None], np.array([k], dtype=np.float64)
        )[0, ...]

    def test_basic(self):
        z = np.array([-1.0, -1.0], dtype=np.complex_)
        p = np.array([0.57149 + 0.29360j, 0.57149 - 0.29360j])
        k = 1
        sos = self.sos2zpk(z, p, k)
        sos2 = [[1, 2, 1, 1, -1.14298, 0.41280]]
        np.testing.assert_array_almost_equal(sos, sos2, decimal=4)

        z = np.array([1j, -1j])
        p = np.array([0.9, -0.9, 0.7j, -0.7j])
        k = 1
        sos = self.sos2zpk(z, p, k)
        sos2 = [[1, 0, 1, 1, 0, +0.49], [1, 0, 0, 1, 0, -0.81]]
        np.testing.assert_array_almost_equal(sos, sos2, decimal=4)

        z = np.array([], dtype=np.complex_)
        p = np.array([0.8, -0.5 + 0.25j, -0.5 - 0.25j])
        k = 1
        sos = self.sos2zpk(z, p, k)
        sos2 = [[1.0, 0.0, 0.0, 1.0, 1.0, 0.3125], [1.0, 0.0, 0.0, 1.0, -0.8, 0.0]]
        np.testing.assert_array_almost_equal(sos, sos2, decimal=4)

        z = np.array([1, 1, 0.9j, -0.9j])
        p = np.array([0.99 + 0.01j, 0.99 - 0.01j, 0.1 + 0.9j, 0.1 - 0.9j])
        k = 1
        sos = self.sos2zpk(z, p, k)
        sos2 = [[1, 0, 0.81, 1, -0.2, 0.82], [1, -2, 1, 1, -1.98, 0.9802]]
        np.testing.assert_array_almost_equal(sos, sos2, decimal=4)

        z = np.array([0.9 + 0.1j, 0.9 - 0.1j, -0.9])
        p = np.array([0.75 + 0.25j, 0.75 - 0.25j, 0.9])
        k = 1
        sos = self.sos2zpk(z, p, k)
        sos2 = [[1, 0.9, 0, 1, -1.5, 0.625], [1, -1.8, 0.82, 1, -0.9, 0]]
        np.testing.assert_array_almost_equal(sos, sos2, decimal=4)
