"""Amplitude demodulation helpers for vibration signals."""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from ._validation import (
    as_finite_1d_array,
    require_positive_finite_scalar,
    require_positive_integer,
)


def _zero_phase_filter(sos, x):
    default_padlen = 3 * (2 * len(sos) + 1)
    return sosfiltfilt(sos, x, padlen=min(default_padlen, len(x) - 1))


def amplitude_demodulate(x, fs, band, lowpass, detector="hilbert",
                         filter_order=4, remove_mean=True):
    """Band-pass, detect and low-pass an amplitude-modulated signal.

    Parameters
    ----------
    x : array_like
        Non-empty, finite, real-valued 1D signal.
    fs : float
        Sampling rate in samples per time unit.
    band : (low, high)
        Carrier band-pass edges in cycles per time unit.
    lowpass : float
        Low-pass cutoff for the demodulated modulation signal.
    detector : {'hilbert', 'square'}, default 'hilbert'
        ``'hilbert'`` returns an amplitude envelope in the same units as x.
        ``'square'`` performs square-law detection and returns squared units.
    filter_order : int, default 4
        Butterworth order used for both zero-phase filters.
    remove_mean : bool, default True
        Remove the demodulated DC component before returning.

    Returns
    -------
    y : ndarray
        Demodulated signal on the original time grid. It can be passed directly
        to ``speed_resample`` or ``speed_resample_aa`` before spectrum analysis.

    Example
    -------
    >>> fs = 4096
    >>> t = np.arange(2 * fs) / fs
    >>> x = (1 + 0.3 * np.cos(2 * np.pi * 20 * t)) * np.cos(2 * np.pi * 500 * t)
    >>> y = amplitude_demodulate(x, fs, band=(400, 600), lowpass=80)
    >>> len(y) == len(x)
    True

    Notes
    -----
    Both filters use forward-backward filtering, so their magnitude responses
    are applied twice and the output has no causal time delay. Edge transients
    should be trimmed when their amplitude matters.
    """
    x = as_finite_1d_array(x, "x", min_size=2, real=True)
    require_positive_finite_scalar(fs, "fs")
    require_positive_finite_scalar(lowpass, "lowpass")
    require_positive_integer(filter_order, "filter_order")
    if detector not in {"hilbert", "square"}:
        raise ValueError("detector must be 'hilbert' or 'square'")
    if not isinstance(remove_mean, (bool, np.bool_)):
        raise ValueError("remove_mean must be boolean")

    try:
        low, high = band
    except (TypeError, ValueError):
        raise ValueError("band must contain exactly two frequency edges") from None
    if np.ndim(low) != 0 or np.ndim(high) != 0:
        raise ValueError("band must contain exactly two scalar frequency edges")
    try:
        valid_band = bool(
            np.isfinite(low) and np.isfinite(high)
            and 0 < low < high < fs / 2
        )
    except (TypeError, ValueError):
        valid_band = False
    if not valid_band:
        raise ValueError("band must satisfy 0 < low < high < fs / 2")
    if lowpass >= fs / 2:
        raise ValueError("lowpass must be below fs / 2")

    band_sos = butter(
        filter_order, (low, high), btype="bandpass", fs=fs, output="sos"
    )
    carrier = _zero_phase_filter(band_sos, x)
    if detector == "hilbert":
        detected = np.abs(hilbert(carrier))
    else:
        detected = carrier**2

    lowpass_sos = butter(
        filter_order, lowpass, btype="lowpass", fs=fs, output="sos"
    )
    y = _zero_phase_filter(lowpass_sos, detected)
    if remove_mean:
        y = y - y.mean()
    return y
