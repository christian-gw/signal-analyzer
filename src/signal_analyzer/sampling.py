"""Uniform-grid polyphase resampling helpers."""

import numpy as np
from scipy.signal import resample_poly

from ._validation import (
    as_finite_1d_array,
    require_positive_finite_scalar,
    require_positive_integer,
)


def _increasing_coordinate(coordinate, name):
    coordinate = as_finite_1d_array(
        coordinate, name=name, min_size=2, real=True
    )
    if np.any(np.diff(coordinate) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return coordinate


def align_to_reference(reference_x, reference_coordinate, x, coordinate):
    """Interpolate ``x`` onto the reference grid within their overlap.

    Both coordinates must be finite, strictly increasing 1D arrays paired
    with equally sized finite 1D signals. The returned coordinate consists of
    reference samples satisfying
    ``max(starts) <= reference_coordinate <= min(ends)``; overlap endpoints
    are therefore included when they are present on the reference grid. At
    least two reference samples must remain.

    This is linear interpolation for explicitly aligning slowly varying
    channels such as speed or process values. It does not apply anti-alias
    filtering and must not be used as a substitute for
    :func:`resample_uniform` when a signal needs decimation.

    Parameters
    ----------
    reference_x, reference_coordinate : array_like
        Reference signal and the grid retained in the output.
    x, coordinate : array_like
        Signal to interpolate and its original coordinate grid.

    Returns
    -------
    aligned_coordinate, reference_aligned, x_aligned : ndarray
        The overlapping portion of the reference grid and both signals on
        that grid. No extrapolation is performed.

    Example
    -------
    >>> reference_coordinate = np.arange(5.0)
    >>> reference_x = reference_coordinate**2
    >>> coordinate = np.array([1.0, 3.0, 5.0])
    >>> x = 2 * coordinate
    >>> common, reference, aligned = align_to_reference(
    ...     reference_x, reference_coordinate, x, coordinate)
    >>> common
    array([1., 2., 3., 4.])
    >>> aligned
    array([2., 4., 6., 8.])
    """
    reference_x = as_finite_1d_array(
        reference_x, name="reference_x", min_size=2
    )
    reference_coordinate = _increasing_coordinate(
        reference_coordinate, "reference_coordinate"
    )
    x = as_finite_1d_array(x, name="x", min_size=2)
    coordinate = _increasing_coordinate(coordinate, "coordinate")

    if reference_x.shape != reference_coordinate.shape:
        raise ValueError(
            "reference_x and reference_coordinate must have equal length"
        )
    if x.shape != coordinate.shape:
        raise ValueError("x and coordinate must have equal length")

    lower = max(reference_coordinate[0], coordinate[0])
    upper = min(reference_coordinate[-1], coordinate[-1])
    overlap = (reference_coordinate >= lower) & (reference_coordinate <= upper)
    if np.count_nonzero(overlap) < 2:
        raise ValueError(
            "coordinate overlap must contain at least two reference samples"
        )

    aligned_coordinate = reference_coordinate[overlap]
    reference_aligned = reference_x[overlap]
    x_aligned = np.interp(aligned_coordinate, coordinate, x)
    return aligned_coordinate, reference_aligned, x_aligned


def resample_uniform(
    x,
    fs,
    up=1,
    down=1,
    axis=-1,
    window=("kaiser", 5.0),
    padtype="line",
):
    """Resample uniformly sampled data by the rational factor ``up / down``.

    Parameters
    ----------
    x : array_like
        Finite real or complex data. The selected axis needs at least two
        samples; other dimensions are preserved.
    fs : float
        Input samples per coordinate unit.
    up, down : int, default 1
        Positive integer interpolation and decimation factors.
    axis : int, default -1
        Axis containing the uniformly sampled coordinate.
    window, padtype
        Forwarded to ``scipy.signal.resample_poly``. The default FIR window and
        line padding provide anti-alias filtering for downsampling.

    Returns
    -------
    y : ndarray
        Resampled data. Its selected-axis length is
        ``ceil(len(x) * up / down)``.
    fs_out : float
        Output samples per coordinate unit, ``fs * up / down``.

    Example
    -------
    >>> t = np.arange(1000) / 1000
    >>> x = np.sin(2 * np.pi * 40 * t)
    >>> y, fs_out = resample_uniform(x, fs=1000, down=4)
    >>> (len(y), fs_out)
    (250, 250.0)
    """
    require_positive_finite_scalar(fs, "fs")
    require_positive_integer(up, "up")
    require_positive_integer(down, "down")

    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, (int, np.integer)
    ):
        raise ValueError("axis must be an integer")
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += x.ndim
    if not 0 <= normalized_axis < x.ndim:
        raise ValueError("axis is out of range for x")
    if x.shape[normalized_axis] < 2:
        raise ValueError("x must contain at least two samples along axis")
    try:
        finite = np.all(np.isfinite(x))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError("x must contain only finite numeric values")

    y = resample_poly(
        x,
        up=int(up),
        down=int(down),
        axis=normalized_axis,
        window=window,
        padtype=padtype,
    )
    return y, float(fs * up / down)
