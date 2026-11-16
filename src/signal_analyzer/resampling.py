"""Speed-to-position resampling with temporal and spatial filtering."""

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from ._validation import (
    require_finite_values,
    require_positive_finite_scalar,
    require_positive_integer,
)


def _validated_resampling_inputs(x, speed, fs, resolution):
    """Return validated arrays shared by the spatial resampling helpers."""
    x = np.asarray(x)
    speed = np.asarray(speed)
    if x.ndim != 1 or speed.ndim != 1 or len(x) != len(speed):
        raise ValueError("x and speed must be 1D arrays of equal length")
    if len(x) < 2:
        raise ValueError("x and speed must contain at least two samples")
    require_finite_values(x, "x")
    require_finite_values(speed, "speed")
    require_positive_finite_scalar(fs, "fs")
    require_positive_finite_scalar(resolution, "resolution")
    if np.any(speed <= 0):
        raise ValueError("speed must be strictly positive")
    return x, speed


def _uniform_position_grid(integrated_end, resolution):
    """Return the stable half-open grid 0 <= position < integrated_end."""
    grid_extent = float(integrated_end * resolution)
    nearest_integer = np.rint(grid_extent)
    tolerance = (
        32 * np.finfo(float).eps * max(1.0, abs(grid_extent))
    )
    if abs(grid_extent - nearest_integer) <= tolerance:
        grid_extent = nearest_integer
    sample_count = int(np.ceil(grid_extent))
    return np.arange(sample_count, dtype=float) / resolution


def max_speed_aa_filter(x, speed, fs, resolution=100, order=8):
    """Low-pass x at the target spatial Nyquist for maximum speed.

    The temporal cutoff is ``max(speed) * resolution / 2``. A zero-phase
    Butterworth filter is applied when this cutoff is below the acquisition
    Nyquist frequency; otherwise a copy of x is returned unchanged.

    This places the fixed temporal cutoff at the most permissive boundary for
    maximum speed. The finite-order response attenuates components near that
    boundary. At lower speeds it does not remove every temporal component above
    the target spatial Nyquist, so it is not a rigorous anti-aliasing solution
    for variable-speed resampling.

    Parameters
    ----------
    x, speed, fs, resolution
        Same units and input contract as :func:`speed_resample`.
    order : int, default 8
        Positive Butterworth filter order.

    Returns
    -------
    y : ndarray
        Temporally filtered signal, with the same shape as x.
    """
    x, speed = _validated_resampling_inputs(x, speed, fs, resolution)
    require_positive_integer(order, "order")

    cutoff = float(np.max(speed) * resolution / 2)
    nyquist = float(fs / 2)
    if cutoff >= nyquist:
        return x.copy()

    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    default_padlen = 3 * (2 * len(sos) + 1)
    padlen = min(default_padlen, len(x) - 1)
    return sosfiltfilt(sos, x, padlen=padlen)


def speed_resample(x, speed, fs, resolution=100):
    """Resample x onto equidistant position coordinates.

    x and speed must be synchronous 1D signals sampled at fs. ``fs`` is
    samples per time unit, and speed must use the same time unit in its
    denominator. For example, ``fs`` in samples/s pairs with speed in rev/s,
    not rpm. Speed may otherwise represent rev/time, length/time, etc.;
    resolution is samples per corresponding integrated unit (rev, length,
    etc.). Units are not parsed or converted automatically.

    Trapezoidal integration estimates position; np.interp performs
    linear interpolation. Requires at least two finite samples and
    strictly positive speed.

    Returns
    -------
    pos, y : ndarray
        Uniform position coordinates and resampled signal. The coordinate grid
        is half-open: ``0 <= pos < integrated final position``. Values within
        floating-point tolerance of an exact grid boundary are treated as that
        boundary, preventing a spurious final sample.

    Example
    -------
    >>> t = np.arange(10000) / 1000
    >>> speed = 10 + 2 * np.sin(2 * np.pi * 0.2 * t)
    >>> position = np.r_[0, np.cumsum((speed[:-1] + speed[1:]) / 2000)]
    >>> x = np.sin(2 * np.pi * 3 * position)
    >>> pos, y = speed_resample(x, speed, fs=1000, resolution=128)
    """
    x, speed = _validated_resampling_inputs(x, speed, fs, resolution)

    pos = np.r_[0, np.cumsum((speed[:-1] + speed[1:]) / (2 * fs))]
    pos_new = _uniform_position_grid(pos[-1], resolution)
    return pos_new, np.interp(pos_new, pos, x)


def speed_resample_aa(x, speed, fs, resolution=100, oversample=4,
                      filter_order=8):
    """Temporal low-pass, then spatial oversampling and FIR decimation.

    First applies :func:`max_speed_aa_filter`, whose fixed temporal cutoff is
    based on maximum speed and the target resolution. Then calls
    speed_resample(..., resolution=resolution*oversample) and decimates with
    scipy.signal.resample_poly using line padding.

    The temporal filter is maximally permissive and therefore cannot suppress
    every spatial alias at lower speeds. The spatial oversampling stage also
    cannot undo aliasing from the original acquisition.

    A useful grid-density check for frequencies up to the nominal max-speed
    cutoff is ``oversample >= max(speed) / min(speed)``. Therefore the default
    4x spatial grid covers speed ratios up to 4:1 by this criterion, although
    equality leaves no interpolation or filter-transition margin. If the
    working band is limited to ``resolution / 2.56``, the corresponding check
    is approximately ``oversample >= 0.78125 * max(speed) / min(speed)``; a
    4:1 ratio then has practical margin. These checks do not constitute a
    guarantee of alias-free nonuniform resampling.

    ``filter_order`` controls the Butterworth order used by
    :func:`max_speed_aa_filter`; it defaults to 8.

    Example
    -------
    >>> t = np.arange(1000) / 1000
    >>> speed = np.full(t.shape, 10.0)
    >>> x = np.sin(2 * np.pi * 3 * speed * t)
    >>> pos, y = speed_resample_aa(x, speed, fs=1000,
    ...                            resolution=128, oversample=4)
    """
    require_positive_integer(oversample, "oversample")
    x_filtered = max_speed_aa_filter(
        x, speed, fs, resolution=resolution, order=filter_order
    )
    pos, y = speed_resample(
        x_filtered, speed, fs, resolution=resolution * oversample
    )
    if len(y) == 1:
        return pos.copy(), y.copy()
    y = resample_poly(y, up=1, down=oversample, padtype="line")
    return pos[::oversample], y
