"""
This module provides functions for extracting coherent eddies from Lagrangian trajectories
by wavelet ridge analysis, following Lilly and Perez-Brunius (2021).

The ridge analysis itself lives in :mod:`clouddrift.ridges`, which is a general method for
modulated oscillations; this module adds the oceanographic layer, converting ridges into
the time-varying ellipses of eddies and their physical properties.

References
----------
Lilly JM, Perez-Brunius P. 2021 Extracting statistically significant eddy signals from large
Lagrangian datasets using wavelet ridge analysis, with application to the Gulf of Mexico.
Nonlin. Processes Geophys. 28, 181-212. (10.5194/npg-28-181-2021).
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from clouddrift.signal import ellipse_circularity, ellipse_parameters

# GOMED variable names, recorded in the attributes so that results can be compared with the
# published Gulf of Mexico Eddy Dataset of Lilly and Perez-Brunius (2021)
_GOMED_NAMES = {
    "kappa": "kappa",
    "circularity": "xi",
    "orientation": "theta",
    "phase": "phi",
    "frequency": "omega",
    "radius": "R",
    "velocity": "V",
}


def eddy_properties(
    displacement_x: NDArray[np.complex128],
    displacement_y: NDArray[np.complex128],
    time: NDArray[np.float64],
) -> xr.Dataset:
    """Return the time-varying ellipse properties of a modulated elliptical signal.

    The input is the pair of analytic displacement signals of an oscillation, which the
    ridge analysis provides as the wavelet transform evaluated along a ridge, Eq. (54) of
    Lilly and Perez-Brunius (2021); see the ``transform`` variable returned by
    :func:`clouddrift.ridges.ridge_analysis`.

    Parameters
    ----------
    displacement_x : NDArray[np.complex128]
        Analytic signal of the eastward displacement of the oscillation, in units of length.
    displacement_y : NDArray[np.complex128]
        Analytic signal of the northward displacement, in the same units.
    time : NDArray[np.float64]
        Times of the samples, in any unit; frequencies are returned in radians per unit of
        time and velocities in units of length per unit of time.

    Returns
    -------
    xr.Dataset
        Ellipse properties along the oscillation, all with dimension ``obs``:

        - ``kappa``: root-mean-square ellipse radius, sqrt((a**2 + b**2)/2), Eq. (5)
        - ``circularity``: signed circularity xi, 1 for a counterclockwise circle, Eq. (5)
        - ``orientation``: orientation theta of the major axis, in radians, Eq. (1)
        - ``phase``: phase phi of the particle on the ellipse, in radians, Eq. (1)
        - ``frequency``: instantaneous frequency omega = phi' + xi * theta', Eqs. (34)-(36)
        - ``radius``: geometric mean radius R = sqrt(a*b), Eq. (71)
        - ``velocity``: kinetic energy velocity V, signed, Eq. (72)

    Notes
    -----
    The kinetic energy velocity is the root-mean-square radius of the ellipse traced by the
    *velocity* signal, not the displacement ellipse, so that ``V**2`` is the kinetic energy
    of the motion. It is obtained here by differentiating the analytic displacement signals
    and applying :func:`clouddrift.signal.ellipse_parameters` to the result. For a circular
    orbit it reduces to ``R * omega``, which is the check applied in the tests.

    Both ``frequency`` and ``velocity`` are obtained by centred differences, of the phase
    and of the displacement respectively, so both carry a relative error of about
    ``(omega * dt)**2 / 6``, where ``dt`` is the sampling interval. That is 4e-5 at 400
    samples per cycle and 4e-3 at 40; for an eddy of a 20-day period sampled hourly it is
    3e-5. The ellipse geometry itself -- ``kappa``, ``circularity``, ``orientation``,
    ``phase`` and ``radius`` -- involves no differencing and is unaffected.

    The instantaneous frequency includes the precession of the ellipse through the
    ``xi * theta'`` term: for an ellipse whose orientation rotates, the frequency is not the
    rate of change of the phase alone.

    Examples
    --------
    From a ridge of a bivariate wavelet transform:

    >>> ridges = ridge_analysis((wx, wy), frequencies, time)
    >>> one = subset(ridges, {"id": 1}, row_dim_name="ridge")
    >>> properties = eddy_properties(one["transform"][0], one["transform"][1], one["time"])

    References
    ----------
    Lilly JM, Perez-Brunius P. 2021 Extracting statistically significant eddy signals from
    large Lagrangian datasets using wavelet ridge analysis, with application to the Gulf of
    Mexico. Nonlin. Processes Geophys. 28, 181-212. (10.5194/npg-28-181-2021).

    See Also
    --------
    :func:`ridge_averaged_properties`, :func:`clouddrift.ridges.ridge_analysis`,
    :func:`clouddrift.signal.ellipse_parameters`
    """
    displacement_x = np.asarray(displacement_x)
    displacement_y = np.asarray(displacement_y)
    time = np.asarray(time, dtype=float)
    if not displacement_x.shape == displacement_y.shape == time.shape:
        raise ValueError(
            "displacement_x, displacement_y and time must have the same shape; got "
            f"{displacement_x.shape}, {displacement_y.shape} and {time.shape}"
        )

    kappa, linearity, orientation, phase = ellipse_parameters(displacement_x, displacement_y)
    circularity = ellipse_circularity(linearity)

    # Instantaneous frequency, Eqs. (34)-(36): the phase rate plus the precession of the
    # ellipse weighted by the circularity
    frequency = np.gradient(phase, time) + circularity * np.gradient(orientation, time)

    # Geometric mean radius, Eq. (71). The semi-axes follow from the linearity, and the
    # product a*b reduces to kappa**2 * |xi|
    radius = kappa * np.sqrt(np.abs(circularity))

    # Kinetic energy velocity, Eq. (72): the signed root-mean-square radius of the ellipse
    # traced by the velocity signal
    velocity_kappa, velocity_linearity, _, _ = ellipse_parameters(
        np.gradient(displacement_x, time), np.gradient(displacement_y, time)
    )
    velocity = np.sign(velocity_linearity) * velocity_kappa

    dataset = xr.Dataset(
        {
            "kappa": (["obs"], kappa),
            "circularity": (["obs"], circularity),
            "orientation": (["obs"], orientation),
            "phase": (["obs"], phase),
            "frequency": (["obs"], frequency),
            "radius": (["obs"], radius),
            "velocity": (["obs"], velocity),
        }
    )
    descriptions = {
        "kappa": ("root-mean-square ellipse radius", "length"),
        "circularity": ("signed ellipse circularity", "dimensionless"),
        "orientation": ("orientation of the ellipse major axis", "radian"),
        "phase": ("phase of the particle on the ellipse", "radian"),
        "frequency": ("instantaneous frequency", "rad/[unit time]"),
        "radius": ("geometric mean radius of the ellipse", "length"),
        "velocity": ("kinetic energy velocity, signed", "length/[unit time]"),
    }
    for name, (long_name, units) in descriptions.items():
        dataset[name].attrs = {
            "long_name": long_name,
            "units": units,
            "gomed_name": _GOMED_NAMES[name],
        }
    return dataset


def ridge_averaged_properties(properties: xr.Dataset) -> xr.Dataset:
    """Return the ridge-averaged properties of a modulated elliptical signal.

    The averages are not all of the same kind, Eqs. (74)-(75) of Lilly and Perez-Brunius
    (2021): the circularity and the nondimensional frequency are averaged with weights
    ``kappa**2``, the instantaneous signal power, while the radius and the kinetic energy
    velocity are root-mean-squares, so that the averaged radius reflects the time-averaged
    area of the ellipse and the averaged velocity its time-averaged kinetic energy. The
    averaged velocity takes its sign from the averaged circularity.

    Parameters
    ----------
    properties : xr.Dataset
        Ellipse properties along one oscillation, as returned by :func:`eddy_properties`.
        Any variable named ``nondimensional_frequency`` is averaged like the circularity.

    Returns
    -------
    xr.Dataset
        The averaged quantities, each a scalar: ``kappa``, ``circularity``, ``frequency``,
        ``radius``, ``velocity``, and ``nondimensional_frequency`` when present.

    See Also
    --------
    :func:`eddy_properties`
    """
    weights = np.asarray(properties["kappa"].values, dtype=float) ** 2
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        raise ValueError("cannot average a ridge whose amplitude is everywhere zero")

    def power_weighted(name: str) -> float:
        return float(np.sum(weights * properties[name].values) / weight_sum)

    def root_mean_square(name: str) -> float:
        return float(np.sqrt(np.mean(properties[name].values ** 2)))

    circularity = power_weighted("circularity")
    averaged = {
        "kappa": root_mean_square("kappa"),
        "circularity": circularity,
        "frequency": power_weighted("frequency"),
        "radius": root_mean_square("radius"),
        "velocity": np.sign(circularity) * root_mean_square("velocity"),
    }
    if "nondimensional_frequency" in properties:
        averaged["nondimensional_frequency"] = power_weighted("nondimensional_frequency")

    dataset = xr.Dataset({name: ((), value) for name, value in averaged.items()})
    for name in dataset.data_vars:
        if name in properties:
            dataset[name].attrs = dict(properties[name].attrs)
        dataset[name].attrs["long_name"] = "ridge-averaged " + dataset[name].attrs.get(
            "long_name", name
        )
    return dataset
