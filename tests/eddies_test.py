import unittest

import numpy as np

from clouddrift.eddies import eddy_properties, ridge_averaged_properties
from clouddrift.signal import ellipse_circularity, modulated_ellipse_signal


def circular_orbit(n_times=1024, radius=20.0, period=400.0, clockwise=False):
    """A circular orbit as a pair of analytic displacement signals."""
    time = np.arange(n_times, dtype=float)
    sign = -1.0 if clockwise else 1.0
    displacement_x = radius * np.exp(1j * sign * 2 * np.pi * time / period)
    displacement_y = -1j * sign * displacement_x
    return time, displacement_x, displacement_y


class eddy_properties_tests(unittest.TestCase):
    def test_circular_orbit(self):
        """For a circle, kappa = R = the radius, |xi| = 1, and V = R * omega."""
        radius, period = 20.0, 400.0
        time, displacement_x, displacement_y = circular_orbit(radius=radius, period=period)
        properties = eddy_properties(displacement_x, displacement_y, time)

        interior = slice(10, -10)  # away from the one-sided gradients at the ends
        omega = 2 * np.pi / period
        np.testing.assert_allclose(properties["kappa"].values[interior], radius)
        np.testing.assert_allclose(properties["circularity"].values[interior], 1.0)
        np.testing.assert_allclose(properties["radius"].values[interior], radius)
        # frequency and velocity are both obtained by centred differences, of the phase
        # and of the displacement, so both carry a relative error of about
        # (omega * dt)**2 / 6, which is 4e-5 at 400 samples per cycle
        np.testing.assert_allclose(properties["frequency"].values[interior], omega, rtol=1e-4)
        np.testing.assert_allclose(
            properties["velocity"].values[interior], radius * omega, rtol=1e-4
        )

    def test_velocity_converges_with_sampling(self):
        """V is obtained by differencing, so its error falls as the sampling refines.

        The centred difference of exp(i*omega*t) has magnitude sin(omega*dt)/dt rather than
        omega, an error of about (omega*dt)**2 / 6. At hourly sampling and an eddy period of
        days that is parts per hundred thousand, but it is worth pinning down.
        """
        radius = 20.0
        errors = []
        for period in (40.0, 80.0):  # halving omega * dt
            time, displacement_x, displacement_y = circular_orbit(radius=radius, period=period)
            properties = eddy_properties(displacement_x, displacement_y, time)
            expected = radius * 2 * np.pi / period
            measured = properties["velocity"].values[10:-10].mean()
            errors.append(abs(measured - expected) / expected)
        # Second-order convergence: the error falls by about four
        self.assertLess(errors[1], errors[0] / 3.5)
        self.assertLess(errors[0], 5e-3)

    def test_rotation_sense(self):
        """Clockwise motion gives a negative circularity and a negative velocity."""
        time, displacement_x, displacement_y = circular_orbit(clockwise=True)
        properties = eddy_properties(displacement_x, displacement_y, time)
        interior = slice(10, -10)
        self.assertTrue(np.all(properties["circularity"].values[interior] < 0))
        self.assertTrue(np.all(properties["velocity"].values[interior] < 0))

    def test_known_ellipse(self):
        """A precessing, amplitude-modulated ellipse is recovered."""
        n_times = 2048
        time = np.arange(n_times, dtype=float)
        kappa = 20.0 * (1 + 0.3 * np.sin(2 * np.pi * time / n_times))
        linearity = np.full(n_times, 0.3)
        orientation = 0.4 * np.pi * time / n_times
        phase_rate = 2 * np.pi / 40.0
        displacement_x, displacement_y = modulated_ellipse_signal(
            kappa, linearity, orientation, phase_rate * time
        )
        properties = eddy_properties(displacement_x, displacement_y, time)

        interior = slice(10, -10)
        circularity = ellipse_circularity(linearity)
        np.testing.assert_allclose(properties["kappa"].values[interior], kappa[interior])
        np.testing.assert_allclose(
            properties["circularity"].values[interior], circularity[interior]
        )
        # The instantaneous frequency includes the precession term, Eqs. (34)-(36)
        expected = phase_rate + circularity * np.gradient(orientation, time)
        np.testing.assert_allclose(
            properties["frequency"].values[interior], expected[interior], rtol=1e-6
        )
        # R = sqrt(a * b), where the semi-axes come from the linearity
        semi_major = kappa * np.sqrt(1 + np.abs(linearity))
        semi_minor = kappa * np.sqrt(1 - np.abs(linearity))
        np.testing.assert_allclose(
            properties["radius"].values[interior],
            np.sqrt(semi_major * semi_minor)[interior],
        )

    def test_units_scale_with_time(self):
        """Frequency and velocity follow the unit of the time argument."""
        time, displacement_x, displacement_y = circular_orbit()
        in_samples = eddy_properties(displacement_x, displacement_y, time)
        in_hours = eddy_properties(displacement_x, displacement_y, time / 24.0)
        interior = slice(10, -10)
        np.testing.assert_allclose(
            in_hours["frequency"].values[interior],
            24.0 * in_samples["frequency"].values[interior],
        )
        np.testing.assert_allclose(
            in_hours["velocity"].values[interior],
            24.0 * in_samples["velocity"].values[interior],
        )
        # Lengths do not
        np.testing.assert_allclose(in_hours["radius"].values, in_samples["radius"].values)

    def test_mismatched_shapes_raise(self):
        time, displacement_x, displacement_y = circular_orbit()
        with self.assertRaises(ValueError):
            eddy_properties(displacement_x, displacement_y[:-1], time)

    def test_gomed_names_recorded(self):
        time, displacement_x, displacement_y = circular_orbit()
        properties = eddy_properties(displacement_x, displacement_y, time)
        self.assertEqual(properties["circularity"].attrs["gomed_name"], "xi")
        self.assertEqual(properties["radius"].attrs["gomed_name"], "R")


class ridge_averaged_properties_tests(unittest.TestCase):
    def test_constant_signal(self):
        """For a steady circular orbit the averages equal the instantaneous values."""
        radius, period = 20.0, 400.0
        time, displacement_x, displacement_y = circular_orbit(radius=radius, period=period)
        properties = eddy_properties(displacement_x, displacement_y, time).isel(obs=slice(10, -10))
        averaged = ridge_averaged_properties(properties)

        self.assertAlmostEqual(float(averaged["kappa"]), radius)
        self.assertAlmostEqual(float(averaged["circularity"]), 1.0)
        self.assertAlmostEqual(float(averaged["radius"]), radius)
        # places=4 acknowledges the centred-difference error in V, (omega*dt)**2 / 6
        self.assertAlmostEqual(float(averaged["velocity"]), radius * 2 * np.pi / period, places=4)

    def test_circularity_is_power_weighted(self):
        """The circularity average is weighted by kappa**2, not taken plainly."""
        n_times = 400
        time = np.arange(n_times, dtype=float)
        # A strong, nearly circular half and a weak, nearly linear half
        kappa = np.where(time < n_times // 2, 10.0, 1.0)
        linearity = np.where(time < n_times // 2, 0.0, 0.99)
        displacement_x, displacement_y = modulated_ellipse_signal(
            kappa, linearity, np.zeros(n_times), 2 * np.pi * time / 40.0
        )
        properties = eddy_properties(displacement_x, displacement_y, time).isel(obs=slice(5, -5))
        averaged = ridge_averaged_properties(properties)

        plain_mean = float(properties["circularity"].mean())
        weights = properties["kappa"].values ** 2
        expected = float(np.sum(weights * properties["circularity"].values) / np.sum(weights))
        self.assertAlmostEqual(float(averaged["circularity"]), expected)
        # The strong circular half dominates, so the weighted average is much larger
        self.assertGreater(float(averaged["circularity"]), plain_mean + 0.05)

    def test_radius_is_root_mean_square(self):
        """R_bar**2 is the time-averaged area, so R_bar is a root-mean-square."""
        n_times = 400
        time = np.arange(n_times, dtype=float)
        kappa = np.where(time < n_times // 2, 30.0, 10.0)
        displacement_x, displacement_y = modulated_ellipse_signal(
            kappa, np.zeros(n_times), np.zeros(n_times), 2 * np.pi * time / 40.0
        )
        properties = eddy_properties(displacement_x, displacement_y, time).isel(obs=slice(5, -5))
        averaged = ridge_averaged_properties(properties)
        expected = np.sqrt(np.mean(properties["radius"].values ** 2))
        self.assertAlmostEqual(float(averaged["radius"]), float(expected))
        self.assertGreater(float(averaged["radius"]), float(properties["radius"].mean()))

    def test_velocity_takes_the_sign_of_the_circularity(self):
        time, displacement_x, displacement_y = circular_orbit(clockwise=True)
        properties = eddy_properties(displacement_x, displacement_y, time).isel(obs=slice(10, -10))
        averaged = ridge_averaged_properties(properties)
        self.assertLess(float(averaged["circularity"]), 0)
        self.assertLess(float(averaged["velocity"]), 0)

    def test_zero_amplitude_raises(self):
        time = np.arange(64, dtype=float)
        zeros = np.zeros(64, dtype=complex)
        properties = eddy_properties(zeros, zeros, time)
        with self.assertRaises(ValueError):
            ridge_averaged_properties(properties)


if __name__ == "__main__":
    unittest.main()
