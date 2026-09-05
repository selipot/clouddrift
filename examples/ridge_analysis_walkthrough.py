"""Step-by-step check of the wavelet ridge analysis in ``clouddrift.ridges``.

Run this as a script, or cell by cell in an interactive window (VS Code: "Run Cell"
above each ``# %%`` marker; Spyder and PyCharm understand the same markers).

Each section prints the numbers it checks, so the output can be read as a report:
sections 1-5 are self-contained and need no data, section 6 needs the GDP hourly
dataset and a network connection.

Contents
--------
1. A synthetic modulated elliptical signal with known parameters
2. Its wavelet transform
3. Joint ridges of the bivariate transform
4. Recovering the analytic signal along the ridge, Eq. (54)
5. Ellipse properties along the ridge, a preview of the eddies module
6. One-sided ridges of a real trajectory, Section 3.7

Reference
---------
Lilly, J. M. and Perez-Brunius, P. (2021). Extracting statistically significant eddy
signals from large Lagrangian datasets using wavelet ridge analysis, with application
to the Gulf of Mexico. Nonlinear Processes in Geophysics, 28, 181-212.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from clouddrift.eddies import eddy_properties, ridge_averaged_properties
from clouddrift.ridges import ridge_analysis
from clouddrift.signal import ellipse_circularity, modulated_ellipse_signal
from clouddrift.wavelet import morse_logspace_freq, morse_wavelet_transform

# Wavelet parameters used throughout, as in Lilly and Perez-Brunius (2021)
GAMMA, BETA = 3, 2
# Minimum ridge length, in number of oscillations: L > n * 2 * P / pi with n = 1,
# where P = sqrt(beta * gamma) is the wavelet duration, their Eq. (61)
MIN_RIDGE_LENGTH = 2 * np.sqrt(BETA * GAMMA) / np.pi
print(f"minimum ridge length: L > {MIN_RIDGE_LENGTH:.2f} oscillations")


# %%
# 1. A synthetic modulated elliptical signal with known parameters
#
# A near-circular, counterclockwise oscillation whose amplitude grows and shrinks and
# whose orientation slowly precesses, on top of a slow background drift.

n_times = 2048
time = np.arange(n_times, dtype=float)  # in samples
period = 40.0  # samples per oscillation
phase_rate = 2 * np.pi / period  # the rate at which the particle goes round the ellipse

kappa_true = 20.0 * (1 + 0.3 * np.sin(2 * np.pi * time / n_times))  # amplitude
lambda_true = np.full(n_times, 0.3)  # linearity, so |xi| = sqrt(1 - 0.09) = 0.954
theta_true = 0.4 * np.pi * time / n_times  # slowly precessing orientation
phi_true = phase_rate * time  # phase of the particle on the ellipse

# The instantaneous frequency of an elliptical signal is NOT the phase rate alone: a
# precessing ellipse contributes too, weighted by the circularity,
#     omega(t) = phi'(t) + xi(t) * theta'(t)
# (Eqs. 34-36 of Lilly and Perez-Brunius 2021). Here the precession adds about 0.4 %, small
# but far larger than the error of the ridge estimate, so comparing the estimate against
# phi' alone would look like a systematic bias of the method when it is nothing of the kind.
omega_true = phase_rate + ellipse_circularity(lambda_true) * np.gradient(theta_true)

xa, ya = modulated_ellipse_signal(kappa_true, lambda_true, theta_true, phi_true)
background_x = 0.02 * time  # a slow drift, the "background process"
background_y = 5.0 * np.sin(2 * np.pi * time / 900.0)
x = np.real(xa) + background_x
y = np.real(ya) + background_y

print(
    f"signal: {n_times} samples, period {period}, true |xi| = "
    f"{np.abs(ellipse_circularity(lambda_true[0])):.3f}"
)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(x, y, lw=0.7)
ax[0].set(xlabel="x", ylabel="y", title="trajectory in the plane", aspect="equal")
ax[1].plot(time, x, lw=0.7, label="x")
ax[1].plot(time, y, lw=0.7, label="y")
ax[1].set(xlabel="time (samples)", title="components")
ax[1].legend()
fig.tight_layout()

# %%
# 2. The wavelet transform of each component
#
# morse_logspace_freq returns radian frequencies from high to low; density sets how
# finely the frequency axis is sampled.

frequencies = morse_logspace_freq(GAMMA, BETA, n_times, density=8)
periods = 2 * np.pi / frequencies

wx = morse_wavelet_transform(x, GAMMA, BETA, frequencies, boundary="mirror")
wy = morse_wavelet_transform(y, GAMMA, BETA, frequencies, boundary="mirror")
transform_norm = np.sqrt(np.abs(wx) ** 2 + np.abs(wy) ** 2)

print(f"transform shape (scale, time): {wx.shape}")
print(f"frequency range: {frequencies.min():.4f} to {frequencies.max():.4f} rad/sample")

fig, ax = plt.subplots(figsize=(11, 4))
mesh = ax.pcolormesh(time, periods, transform_norm, shading="auto", cmap="viridis")
ax.axhline(period, color="w", ls="--", lw=1, label="true period")
ax.set(
    yscale="log",
    xlabel="time (samples)",
    ylabel="period (samples)",
    title=r"$\|w(t,s)\|$, norm of the bivariate transform",
)
ax.invert_yaxis()
ax.legend()
fig.colorbar(mesh, ax=ax, label="transform norm")
fig.tight_layout()

# %%
# 3. Joint ridges of the bivariate transform
#
# The transform is passed as a pair, so ridges are joint ridges of ||w(t,s)||: local
# maxima with respect to scale, connected in time. This is the multivariate ridge
# definition of Lilly and Olhede (2012), their Eq. (52) in the 2021 paper.

ridges = ridge_analysis((wx, wy), frequencies, time, min_ridge_size=20)

rowsize = ridges["rowsize"].values
start = np.concatenate([[0], np.cumsum(rowsize)])
lengths = ridges["ridge_length"].values

print(f"{ridges.sizes['ridge']} ridges, {ridges.sizes['obs']} ridge points")
print(f"{'ridge':>6} {'points':>7} {'L':>8} {'mean period':>12}")
for k, ridge_id in enumerate(ridges["id"].values):
    ridge = slice(start[k], start[k + 1])
    mean_period = 2 * np.pi / ridges["inst_frequency"].values[ridge].mean()
    flag = "  <- kept" if lengths[k] > MIN_RIDGE_LENGTH else ""
    print(f"{int(ridge_id):>6} {rowsize[k]:>7} {lengths[k]:>8.2f} {mean_period:>12.2f}{flag}")

# The result follows the CloudDrift ragged convention: an observation dimension "obs", a
# row dimension "ridge", and an "id" variable naming each row, so a single ridge comes out
# with subset and its defaults
from clouddrift.ragged import subset  # noqa: E402

first = subset(ridges, {"id": int(ridges["id"].values[0])}, row_dim_name="ridge")
print(
    f"\nsubset of ridge {int(ridges['id'].values[0])}: "
    f"{first.sizes['obs']} points, L = {float(first['ridge_length'].values[0]):.2f}"
)

# The longest ridge is the oscillation; the rest are short, spurious features that the
# ridge length criterion removes
longest = int(np.argmax(lengths))
main = slice(start[longest], start[longest + 1])
ridge_time = ridges["time"].values[main]
# L is the integral of the instantaneous frequency, Eq. (60), so the precession enters here
# too: the expected number of oscillations is not simply the record length over the period
expected_length = np.trapezoid(omega_true, time) / (2 * np.pi)
print(
    f"\nlongest ridge: L = {lengths[longest]:.2f}, expected {expected_length:.2f} oscillations "
    f"({n_times / period:.2f} from the phase rate alone)"
)

# %%
# 4. Recovering the analytic signal along the ridge
#
# Evaluating the transform along the ridge estimates the analytic signal of the
# modulated oscillation, Eq. (54). ridge_analysis returns it as the "transform"
# variable, one row per component.

x_star = ridges["transform"].values[0][main]
y_star = ridges["transform"].values[1][main]

# Compare with the truth, away from the edges where the wavelet runs off the record
interior = (ridge_time > 200) & (ridge_time < n_times - 200)
index = ridge_time[interior].astype(int)

amplitude_error = np.abs(np.abs(x_star[interior]) - np.abs(xa[index])) / np.abs(xa[index]).max()
estimated_frequency = ridges["inst_frequency"].values[main][interior]
frequency_error = np.abs(estimated_frequency - omega_true[index]) / omega_true[index]
phase_rate_error = np.abs(estimated_frequency - phase_rate) / phase_rate

print(
    f"amplitude of x*, relative error: median {np.median(amplitude_error):.2%}, "
    f"max {amplitude_error.max():.2%}"
)
print(
    f"instantaneous frequency, relative error: median {np.median(frequency_error):.3%}, "
    f"max {frequency_error.max():.3%}"
)
print(
    f"   the same estimate against the phase rate alone: {np.median(phase_rate_error):.3%} - "
    "not a bias of the method, but the precession term it correctly includes"
)

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(time, x, lw=0.6, color="0.7", label="observed x")
ax[0].plot(ridge_time, np.real(x_star), lw=1, label="x* along the ridge")
ax[0].plot(time, np.real(xa), lw=0.8, ls="--", label="true oscillation")
ax[0].set(ylabel="x")
ax[0].legend(ncol=3, fontsize=8)
ax[1].plot(ridge_time, ridges["inst_frequency"].values[main], label="instantaneous frequency")
ax[1].plot(ridge_time, ridges["frequency"].values[main], lw=0.8, label="scale frequency")
ax[1].plot(time, omega_true, "k--", lw=0.8, label=r"true $\omega = \phi' + \xi\theta'$")
ax[1].axhline(phase_rate, color="0.6", ls=":", lw=0.8, label=r"phase rate $\phi'$ alone")
ax[1].set(xlabel="time (samples)", ylabel="rad/sample")
ax[1].legend(ncol=3, fontsize=8)
fig.tight_layout()

# %%
# 5. Ellipse properties along the ridge
#
# clouddrift.eddies.eddy_properties turns the pair of analytic signals into the
# time-varying ellipse of Lilly and Perez-Brunius (2021) and the eddy properties that
# follow from it. Note that the semi-axes come from the LINEARITY,
# a = kappa * sqrt(1 + |lambda|), while the geometric mean radius reduces to
# R = kappa * sqrt(|xi|) in terms of the CIRCULARITY: mixing the two gives a plausible
# but wrong radius, which is why the identity is checked below.

properties = eddy_properties(x_star, y_star, ridge_time)
kappa = properties["kappa"].values
xi = properties["circularity"].values
theta = properties["orientation"].values
radius = properties["radius"].values
velocity = properties["velocity"].values

xi_true = ellipse_circularity(lambda_true[0])
print(f"kappa: recovered {kappa[interior].mean():.2f}, true {kappa_true[index].mean():.2f}")
print(f"xi:    recovered {xi[interior].mean():.3f}, true {xi_true:.3f}")
radius_true = kappa_true[index] * np.sqrt(np.abs(ellipse_circularity(lambda_true[index])))
print(f"R:     recovered {radius[interior].mean():.2f}, true {radius_true.mean():.2f}")
print(
    "       consistency, R vs kappa * sqrt(|xi|): "
    f"{np.abs(radius[interior] - kappa[interior] * np.sqrt(np.abs(xi[interior]))).max():.2e}"
)
# V is the kinetic energy velocity, the signed root-mean-square radius of the ellipse
# traced by the VELOCITY signal, Eq. (72). For a circular orbit it would equal R * omega;
# for this ellipse it is close to that but not equal to it.
print(
    f"V:     recovered {velocity[interior].mean():.3f}, "
    f"R * omega would give {(radius[interior] * omega_true[index]).mean():.3f}"
)
print(f"rotation sense: {'counterclockwise' if xi[interior].mean() > 0 else 'clockwise'}")

# The ridge-averaged quantities, Eqs. (74)-(75): xi is weighted by the signal power
# kappa**2, while R and V are root-mean-squares, so that R**2 is the time-averaged area
# and V**2 the time-averaged kinetic energy
averaged = ridge_averaged_properties(properties.isel(obs=interior))
print(
    f"ridge averages: kappa {float(averaged['kappa']):.2f}, xi {float(averaged['circularity']):.3f}, "
    f"R {float(averaged['radius']):.2f}, V {float(averaged['velocity']):.3f}"
)

fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
ax[0].plot(ridge_time, kappa, label="recovered")
ax[0].plot(time, kappa_true, ls="--", lw=0.8, label="true")
ax[0].set(ylabel=r"$\kappa$")
ax[0].legend(fontsize=8)
ax[1].plot(ridge_time, xi)
ax[1].axhline(xi_true, color="k", ls="--", lw=0.8)
ax[1].set(ylabel=r"circularity $\xi$", ylim=(-1.05, 1.05))
ax[2].plot(ridge_time, np.unwrap(theta), label="recovered")
ax[2].plot(time, theta_true, ls="--", lw=0.8, label="true")
ax[2].set(ylabel=r"orientation $\theta$")
ax[2].legend(fontsize=8)
ax[3].plot(ridge_time, radius, label="R")
ax[3].plot(ridge_time, velocity, label="V")
ax[3].set(xlabel="time (samples)", ylabel="R, V")
ax[3].legend(fontsize=8)
fig.tight_layout()

# %%
# 6. One-sided ridges of a real trajectory
#
# Needs the GDP hourly dataset, so this section reaches the network. The transform is
# taken on the sphere with a moving tangent plane, then split into rotary components:
# masking on the sign of |w+| - |w-| prevents a ridge from crossing between cyclonic
# and anticyclonic polarizations, Section 3.7.

import pathlib  # noqa: E402

import xarray as xr  # noqa: E402

from clouddrift.datasets import gdp1h  # noqa: E402
from clouddrift.ragged import subset  # noqa: E402
from clouddrift.sphere import EARTH_RADIUS_METERS, coriolis_frequency  # noqa: E402
from clouddrift.wavelet import morse_spherical_wavelet_transform  # noqa: E402

# A local copy of this one drifter, if it has been saved beside the repository, so the
# section can run without network access; otherwise the dataset is opened remotely. Run
# the script from the repository root for the relative path to resolve.
LOCAL_DRIFTER = pathlib.Path("data/gdp1h_44000.nc")
if LOCAL_DRIFTER.exists():
    drifter = xr.open_dataset(LOCAL_DRIFTER)
    print(f"using the local copy at {LOCAL_DRIFTER}")
else:
    drifter = subset(gdp1h(), {"id": [44000]}, row_dim_name="traj")
latitude = drifter["lat"].values
longitude = drifter["lon"].values
# Time in hours since the start of the record
hours = (drifter["time"].values - drifter["time"].values[0]) / np.timedelta64(1, "h")

# The wavelet transform assumes uniform sampling, so any gap is worth knowing about
steps = np.diff(hours)
if not np.allclose(steps, steps[0]):
    gaps = np.flatnonzero(steps > 1.5 * np.median(steps))
    print(
        f"sampling: median {np.median(steps):.2f} h, {len(gaps)} gap(s) longer than that, "
        f"the largest {steps[gaps].max():.0f} h at day {hours[gaps[np.argmax(steps[gaps])]] / 24:.0f}. "
        "Gaps are treated here as if the record were continuous."
    )

frequencies_gdp = morse_logspace_freq(GAMMA, BETA, len(hours), density=4)
periods_gdp = 2 * np.pi / frequencies_gdp / 24  # in days

# Kilometres, so that radii come out in kilometres
wx_gdp, wy_gdp = morse_spherical_wavelet_transform(
    latitude,
    longitude,
    GAMMA,
    BETA,
    frequencies_gdp,
    boundary="mirror",
    radius=EARTH_RADIUS_METERS / 1e3,
)

# Rotary transforms and the polarization difference
w_plus = (wx_gdp + 1j * wy_gdp) / np.sqrt(2)
w_minus = (wx_gdp - 1j * wy_gdp) / np.sqrt(2)
transform_difference = np.abs(w_plus) - np.abs(w_minus)

# The same joint ridge analysis, run twice under complementary masks. The names below
# assume the Northern Hemisphere, where positive (counterclockwise) rotation is cyclonic;
# in general the sense follows the sign of the Coriolis frequency, which is why the paper
# works with omega_ast = sign(xi) * omega / f rather than with the rotation sense alone.
cyclonic = ridge_analysis(
    (wx_gdp, wy_gdp),
    frequencies_gdp,
    hours,
    mask=transform_difference >= 0,
    min_ridge_size=24,
)
anticyclonic = ridge_analysis(
    (wx_gdp, wy_gdp),
    frequencies_gdp,
    hours,
    mask=transform_difference < 0,
    min_ridge_size=24,
)

for name, result in [("cyclonic", cyclonic), ("anticyclonic", anticyclonic)]:
    kept = result["ridge_length"].values > MIN_RIDGE_LENGTH
    print(
        f"{name:>13}: {result.sizes['ridge']:>3} ridges, {kept.sum():>3} longer than "
        f"{MIN_RIDGE_LENGTH:.2f} oscillations"
    )

# %%
# Ridges over the transform, with the local inertial period for reference

coriolis_period = 2 * np.pi / np.abs(coriolis_frequency(latitude) * 3600) / 24  # days
norm_gdp = np.sqrt(np.abs(wx_gdp) ** 2 + np.abs(wy_gdp) ** 2)

fig, ax = plt.subplots(figsize=(12, 5))
mesh = ax.pcolormesh(hours / 24, periods_gdp, np.log10(norm_gdp), shading="auto", cmap="Greys")
ax.plot(hours / 24, coriolis_period, "k--", lw=1, label="inertial period")

for result, color, name in [
    (cyclonic, "tab:red", "cyclonic"),
    (anticyclonic, "tab:blue", "anticyclonic"),
]:
    sizes = result["rowsize"].values
    edges = np.concatenate([[0], np.cumsum(sizes)])
    labelled = False
    for k in range(result.sizes["ridge"]):
        if result["ridge_length"].values[k] <= MIN_RIDGE_LENGTH:
            continue
        ridge = slice(edges[k], edges[k + 1])
        ax.plot(
            result["time"].values[ridge] / 24,
            2 * np.pi / result["frequency"].values[ridge] / 24,
            color=color,
            lw=1.5,
            label=None if labelled else name,
        )
        labelled = True

ax.set(
    yscale="log",
    xlabel="time (days)",
    ylabel="period (days)",
    title="One-sided ridges, GDP drifter 44000",
)
ax.invert_yaxis()
ax.legend()
fig.colorbar(mesh, ax=ax, label=r"$\log_{10}\|w\|$ (km)")
fig.tight_layout()
plt.show()


# %%
# 7. Inertial oscillations shifted by a cyclone
#
# Inertial oscillations inside an eddy are shifted from f to f + zeta/2 (Kunze, 1985),
# and for a core in solid-body rotation zeta/2 is the angular velocity of the eddy, so the
# shifted frequency is f + omega_eddy: the frequency of the cyclonic ridge adds to the
# frequency of the anticyclonic inertial ridge during the eddy.
#
# Detecting both at once is demanding. A ridge point is a local maximum of ||w(t,s)|| with
# respect to scale, and with beta = 2 the wavelet is short in time and correspondingly
# broad in frequency, so an energetic eddy spreads enough power into the inertial band to
# flatten the weaker inertial maximum. On synthetic tests with a 0.5 km inertial signal
# under an 80 km eddy, detection inside the eddy falls to 7 % of the time while remaining
# at 70 % outside it. Two changes each restore it, and this section runs both:
#
#   - a larger beta narrows the wavelet in frequency, separating the eddy from the
#     inertial band (97 % inside the eddy at beta = 5, 99 % at beta = 8). Section 3.6 of
#     the paper notes the tradeoff: small beta suits strongly modulated signals, large beta
#     resolves closely spaced oscillations, which is what is needed here;
#   - detecting ridges on the negative rotary transform alone rather than on the joint
#     bivariate norm keeps the eddy, which is almost purely positively rotating, out of the
#     competition entirely (94 % inside the eddy even at beta = 2).
#
# Raising amplitude_threshold does not help: it is a floor on |w|, and the obstruction is
# the shape of ||w|| in scale, not its level.

BETA_NARROW = 8
frequencies_narrow = morse_logspace_freq(GAMMA, BETA_NARROW, len(hours), density=4)
wx_narrow, wy_narrow = morse_spherical_wavelet_transform(
    latitude,
    longitude,
    GAMMA,
    BETA_NARROW,
    frequencies_narrow,
    boundary="mirror",
    radius=EARTH_RADIUS_METERS / 1e3,
)
difference_narrow = np.abs(wx_narrow + 1j * wy_narrow) - np.abs(wx_narrow - 1j * wy_narrow)

cyclonic_narrow = ridge_analysis(
    (wx_narrow, wy_narrow),
    frequencies_narrow,
    hours,
    mask=difference_narrow >= 0,
    min_ridge_size=24,
)
anticyclonic_narrow = ridge_analysis(
    (wx_narrow, wy_narrow),
    frequencies_narrow,
    hours,
    mask=difference_narrow < 0,
    min_ridge_size=24,
)
print(
    f"beta = {BETA_NARROW}: {cyclonic_narrow.sizes['ridge']} cyclonic and "
    f"{anticyclonic_narrow.sizes['ridge']} anticyclonic ridges "
    f"(beta = {BETA} gave {cyclonic.sizes['ridge']} and {anticyclonic.sizes['ridge']})"
)

# Choosing the eddy ridge needs care. The ridge length L counts oscillations, so the
# longest ridge by L is typically a high-frequency one: ranking by L alone selects a
# near-inertial cyclonic ridge, and the resulting f + omega then lands near half the
# inertial period, nowhere near an eddy. An eddy is a SUB-INERTIAL feature, turning more
# slowly than the inertial oscillation, so candidates are restricted by frequency first
# and then ranked by how long they last.
coriolis_hourly = np.abs(coriolis_frequency(latitude) * 3600)  # rad/hour
median_coriolis = np.median(coriolis_hourly)

# The eddies are taken from the beta = 2 analysis, which resolves their start and end
# times: at beta = 8 the wavelet spans about five oscillations, which at a 20-day period is
# a hundred days, so low-frequency ridges are badly smeared in time. The near-inertial
# ridges come from the beta = 8 analysis, which is what makes them detectable inside an
# eddy at all. Using each resolution where it is strong is the point of running both.
edges = np.concatenate([[0], np.cumsum(cyclonic["rowsize"].values)])
candidates = []
for k in range(cyclonic.sizes["ridge"]):
    ridge = slice(edges[k], edges[k + 1])
    ridge_time = cyclonic["time"].values[ridge]
    frequency = np.abs(cyclonic["inst_frequency"].values[ridge])
    if cyclonic["ridge_length"].values[k] <= MIN_RIDGE_LENGTH:
        continue
    if np.median(frequency) > 0.5 * median_coriolis:  # sub-inertial only
        continue
    candidates.append((ridge_time[-1] - ridge_time[0], k, ridge_time, frequency))

candidates.sort(reverse=True, key=lambda candidate: candidate[0])
print(
    f"{len(candidates)} sub-inertial cyclonic ridges longer than {MIN_RIDGE_LENGTH:.2f} "
    "oscillations; the five longest-lived:"
)
for duration, k, ridge_time, frequency in candidates[:5]:
    print(
        f"   days {ridge_time[0] / 24:6.0f} to {ridge_time[-1] / 24:6.0f} "
        f"({duration / 24:5.0f} days), "
        f"period {2 * np.pi / np.median(frequency) / 24:6.1f} days, "
        f"L = {cyclonic['ridge_length'].values[k]:5.1f}"
    )

# Rather than singling out one eddy, every sub-inertial cyclonic ridge gets its own
# prediction, and the anticyclonic ridge points that coexist with it in time are compared
# against both the local inertial frequency and the shifted one. This turns the figure
# into a measurement: if the shift is real, the measured frequency should sit closer to
# f + omega_eddy than to f.
anticyclonic_edges = np.concatenate([[0], np.cumsum(anticyclonic_narrow["rowsize"].values)])
anticyclonic_time = anticyclonic_narrow["time"].values
anticyclonic_frequency = np.abs(anticyclonic_narrow["inst_frequency"].values)
keep = np.zeros(anticyclonic_time.size, dtype=bool)
for k in range(anticyclonic_narrow.sizes["ridge"]):
    if anticyclonic_narrow["ridge_length"].values[k] > MIN_RIDGE_LENGTH:
        keep[anticyclonic_edges[k] : anticyclonic_edges[k + 1]] = True

# Comparing medians of periods would conflate the shift with the drifter's latitude, which
# here runs from 19 to 46 degrees, so f itself varies by a factor of two. The comparison is
# made point by point instead, as the ratio of the measured frequency to each prediction:
# a ratio of one means the prediction is right.
predictions = []
print("\nnear-inertial anticyclonic ridges during each cyclonic ridge, point by point:")
print(
    f"   {'eddy days':>16} {'eddy period':>10} {'omega/f':>8} {'points':>7} "
    f"{'measured/f':>11} {'measured/(f+omega)':>19}  verdict"
)
for duration, k, ridge_time, frequency in candidates:
    if duration < 20 * 24:  # ignore ridges lasting less than 20 days
        continue
    eddy_on_full_axis = np.interp(hours, ridge_time, frequency, left=0.0, right=0.0)
    predictions.append(eddy_on_full_axis)

    # anticyclonic ridge points inside this eddy and within a broad inertial band
    coriolis_at_point = np.interp(anticyclonic_time, hours, coriolis_hourly)
    shift_at_point = np.interp(anticyclonic_time, hours, eddy_on_full_axis)
    near_inertial = (
        keep
        & (shift_at_point > 0)
        & (anticyclonic_frequency > 0.6 * coriolis_at_point)
        & (anticyclonic_frequency < 2.0 * coriolis_at_point)
    )
    if near_inertial.sum() < 24:
        continue
    measured = anticyclonic_frequency[near_inertial]
    plain = coriolis_at_point[near_inertial]
    shifted = plain + shift_at_point[near_inertial]
    relative_shift = np.median(shift_at_point[near_inertial] / plain)
    ratio_plain = np.median(measured / plain)
    ratio_shifted = np.median(measured / shifted)
    if relative_shift < 0.05:
        verdict = "shift too small to resolve"
    elif abs(ratio_shifted - 1) < abs(ratio_plain - 1):
        verdict = "closer to f + omega"
    else:
        verdict = "closer to f"
    print(
        f"   {ridge_time[0] / 24:6.0f} to {ridge_time[-1] / 24:5.0f} "
        f"{2 * np.pi / np.median(frequency) / 24:8.1f} d {relative_shift:8.2f} "
        f"{near_inertial.sum():7d} {ratio_plain:11.3f} {ratio_shifted:19.3f}  {verdict}"
    )

# The prediction curves, one per eddy, for the figure below
shifted_inertial_periods = [
    (2 * np.pi / (coriolis_hourly + prediction) / 24, prediction > 0) for prediction in predictions
]

# %%
norm_narrow = np.sqrt(np.abs(wx_narrow) ** 2 + np.abs(wy_narrow) ** 2)
periods_narrow = 2 * np.pi / frequencies_narrow / 24

fig, ax = plt.subplots(figsize=(12, 5))
mesh = ax.pcolormesh(
    hours / 24, periods_narrow, np.log10(norm_narrow), shading="auto", cmap="Greys"
)
ax.plot(hours / 24, coriolis_period, "k--", lw=1, label="inertial period")

for result, color, name in [
    (cyclonic_narrow, "tab:red", "cyclonic"),
    (anticyclonic_narrow, "tab:blue", "anticyclonic"),
]:
    sizes = result["rowsize"].values
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    labelled = False
    for k in range(result.sizes["ridge"]):
        if result["ridge_length"].values[k] <= MIN_RIDGE_LENGTH:
            continue
        ridge = slice(bounds[k], bounds[k + 1])
        ax.plot(
            result["time"].values[ridge] / 24,
            2 * np.pi / result["frequency"].values[ridge] / 24,
            color=color,
            lw=1.5,
            label=None if labelled else name,
        )
        labelled = True

for index, (period_curve, inside) in enumerate(shifted_inertial_periods):
    ax.plot(
        hours[inside] / 24,
        period_curve[inside],
        color="magenta",
        lw=1.5,
        label=r"$2\pi/(f + \omega_{eddy})$" if index == 0 else None,
    )

ax.set(
    yscale="log",
    xlabel="time (days)",
    ylabel="period (days)",
    title=f"One-sided ridges with beta = {BETA_NARROW}, GDP drifter 44000",
)
ax.invert_yaxis()
ax.legend(loc="lower left")
fig.colorbar(mesh, ax=ax, label=r"$\log_{10}\|w\|$ (km)")
fig.tight_layout()
plt.show()

# %%
# 8. Eddy properties of a real feature
#
# The property layer applied to the cyclone found in section 7, giving the quantities that
# a census would report: the geometric mean radius R, the kinetic energy velocity V, the
# circularity xi, and the nondimensional frequency omega_ast = sign(xi) * omega / f, whose
# double is the vortex Rossby number, Eq. (67).

eddy_edges = np.concatenate([[0], np.cumsum(cyclonic["rowsize"].values)])
_, eddy_index, eddy_time, _ = candidates[0]
eddy_slice = slice(eddy_edges[eddy_index], eddy_edges[eddy_index + 1])

# The transform along the ridge is the analytic displacement signal of the eddy, in km
displacement_x = cyclonic["transform"].values[0][eddy_slice]
displacement_y = cyclonic["transform"].values[1][eddy_slice]
eddy = eddy_properties(displacement_x, displacement_y, eddy_time)

# The centre of the oscillation is the observed position minus the eddy displacement,
# Eqs. (55)-(58), converted from kilometres to degrees
observation = eddy_time.astype(int)
earth_radius_km = EARTH_RADIUS_METERS / 1e3
latitude_offset = np.real(displacement_y) / earth_radius_km * 180 / np.pi
longitude_offset = (
    np.real(displacement_x)
    / (earth_radius_km * np.cos(np.radians(latitude[observation])))
    * 180
    / np.pi
)
centre_latitude = latitude[observation] - latitude_offset
centre_longitude = longitude[observation] - longitude_offset

# Nondimensional frequency at the centre of the oscillation, Eq. (63)
coriolis_at_centre = np.abs(coriolis_frequency(centre_latitude) * 3600)  # rad/hour
nondimensional_frequency = (
    np.sign(eddy["circularity"].values) * eddy["frequency"].values / coriolis_at_centre
)
eddy["nondimensional_frequency"] = (["obs"], nondimensional_frequency)
averaged_eddy = ridge_averaged_properties(eddy)

KM_PER_HOUR_TO_CM_PER_S = 1e5 / 3600
print(
    f"\neddy ridge, days {eddy_time[0] / 24:.0f} to {eddy_time[-1] / 24:.0f}, "
    f"centred near {np.median(centre_latitude):.1f} N, {np.median(centre_longitude):.1f} E"
)
# Both the ridge average and the median of the instantaneous values are printed, because
# they are not the same statistic and can differ a lot. The averages of Eqs. (74)-(75) are
# weighted by the signal power kappa**2 for xi and omega_ast, and are root-mean-squares for
# R and V, so a ridge whose frequency drifts is reported at the frequency of its strongest,
# largest part rather than at its typical one.
print(f"   {'quantity':<22} {'ridge average':>14} {'median':>9}")
print(
    f"   {'period (days)':<22} {2 * np.pi / float(averaged_eddy['frequency']) / 24:14.2f} "
    f"{2 * np.pi / np.median(eddy['frequency'].values) / 24:9.2f}"
)
print(
    f"   {'radius R (km)':<22} {float(averaged_eddy['radius']):14.1f} "
    f"{np.median(eddy['radius'].values):9.1f}"
)
print(
    f"   {'velocity V (cm/s)':<22} "
    f"{float(averaged_eddy['velocity']) * KM_PER_HOUR_TO_CM_PER_S:14.1f} "
    f"{np.median(eddy['velocity'].values) * KM_PER_HOUR_TO_CM_PER_S:9.1f}"
)
print(
    f"   {'circularity xi':<22} {float(averaged_eddy['circularity']):14.2f} "
    f"{np.median(eddy['circularity'].values):9.2f}"
)
print(
    f"   {'omega_ast':<22} {float(averaged_eddy['nondimensional_frequency']):14.2f} "
    f"{np.median(nondimensional_frequency):9.2f}"
)
print(
    f"   {'vortex Rossby number':<22} "
    f"{2 * float(averaged_eddy['nondimensional_frequency']):14.2f} "
    f"{2 * np.median(nondimensional_frequency):9.2f}"
)

# %%
# The eddy ellipses along the trajectory, one per period, as in Fig. 1b of the paper

period_hours = 2 * np.pi / eddy["frequency"].values
snapshots = [0]
while snapshots[-1] + period_hours[snapshots[-1]] < len(eddy_time) - 1:
    snapshots.append(int(snapshots[-1] + period_hours[snapshots[-1]]))

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(
    longitude[observation], latitude[observation], color="0.7", lw=0.6, label="observed trajectory"
)
ax.plot(centre_longitude, centre_latitude, color="k", lw=1, label="estimated eddy centre")

angle = np.linspace(0, 2 * np.pi, 100)
for index, snapshot in enumerate(snapshots):
    kappa = eddy["kappa"].values[snapshot]
    xi = eddy["circularity"].values[snapshot]
    orientation = eddy["orientation"].values[snapshot]
    linearity = np.sqrt(max(1 - xi**2, 0.0))
    semi_major = kappa * np.sqrt(1 + linearity)
    semi_minor = np.sign(xi) * kappa * np.sqrt(1 - linearity)
    offset_x = semi_major * np.cos(angle) * np.cos(orientation) - semi_minor * np.sin(
        angle
    ) * np.sin(orientation)
    offset_y = semi_major * np.cos(angle) * np.sin(orientation) + semi_minor * np.sin(
        angle
    ) * np.cos(orientation)
    ax.plot(
        centre_longitude[snapshot]
        + offset_x
        / (earth_radius_km * np.cos(np.radians(centre_latitude[snapshot])))
        * 180
        / np.pi,
        centre_latitude[snapshot] + offset_y / earth_radius_km * 180 / np.pi,
        color="tab:red",
        lw=1,
        label="eddy ellipse, one per period" if index == 0 else None,
    )

ax.set(
    xlabel="longitude",
    ylabel="latitude",
    title=f"Eddy ellipses, drifter 44000, days {eddy_time[0] / 24:.0f}-{eddy_time[-1] / 24:.0f}",
)
ax.set_aspect(1 / np.cos(np.radians(np.median(centre_latitude))))
ax.legend(fontsize=8)
fig.tight_layout()

# %%
# The properties as time series

fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
ax[0].plot(eddy_time / 24, eddy["radius"].values)
ax[0].set(ylabel="R (km)")
ax[1].plot(eddy_time / 24, eddy["velocity"].values * KM_PER_HOUR_TO_CM_PER_S)
ax[1].set(ylabel="V (cm/s)")
ax[2].plot(eddy_time / 24, eddy["circularity"].values)
ax[2].axhline(0, color="0.7", lw=0.8)
ax[2].set(ylabel=r"circularity $\xi$", ylim=(-1.05, 1.05))
ax[3].plot(eddy_time / 24, nondimensional_frequency)
ax[3].axhline(0.5, color="0.7", ls="--", lw=0.8)
ax[3].set(xlabel="time (days)", ylabel=r"$\omega_*$")
fig.suptitle("Eddy properties along the ridge")
fig.tight_layout()
plt.show()
