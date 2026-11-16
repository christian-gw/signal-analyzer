"""Small FFT helpers for exploratory vibration and order analysis."""

import numpy as np
from scipy.signal import get_window, periodogram

from ._validation import (
    as_finite_1d_array,
    require_positive_finite_scalar,
    require_positive_integer,
)


def fft_spectrum(x, fs=1.0, window="hann", one_sided=True,
                 window_correction=True):
    """Return frequency bins and the complex, amplitude-normalized FFT.

    Parameters
    ----------
    x : array_like
        Non-empty, finite 1D signal (real-valued when one_sided=True).
    fs : float, default 1.0
        Samples per time unit or per spatial unit. Returned frequencies are
        cycles per the corresponding unit.
    window : str, tuple or None, default "hann"
        SciPy window specification; None uses a rectangular window.
    one_sided : bool, default True
        Use rFFT and double interior bins (not DC/Nyquist).
    window_correction : bool, default True
        Divide by the coherent window gain (sum of window weights).
        If False, divide by the number of samples.

    Returns
    -------
    f, Xf : ndarray
        Frequency bins and complex Fourier coefficients.

    Example
    -------
    >>> t = np.arange(1000) / 1000
    >>> x = 2 * np.sin(2 * np.pi * 50 * t)
    >>> f, Xf = fft_spectrum(x, **{"fs": 1000, "window": "hann"})
    >>> round(float(abs(Xf[np.argmax(abs(Xf))])), 4)
    2.0

    Notes
    -----
    Amplitude normalization, *not* PSD normalization.
    """
    x = as_finite_1d_array(x)
    require_positive_finite_scalar(fs, "fs")
    if one_sided and np.iscomplexobj(x):
        raise ValueError("one_sided=True requires real-valued x")

    n = len(x)
    w = get_window(window, n) if window is not None else np.ones(n)

    if one_sided:
        f = np.fft.rfftfreq(n, d=1 / fs)
        Xf = np.fft.rfft(x * w)
    else:
        f = np.fft.fftfreq(n, d=1 / fs)
        Xf = np.fft.fft(x * w)

    Xf /= w.sum() if window_correction else n
    if one_sided:
        Xf[1:(-1 if n % 2 == 0 else None)] *= 2

    return f, Xf


def psd_spectrum(x, fs=1.0, window="hann", one_sided=True,
                 detrend=False):
    """Return frequency bins and a power spectral density estimate.

    Parameters
    ----------
    x : array_like
        Non-empty, finite 1D signal (real-valued when one_sided=True).
    fs : float, default 1.0
        Samples per time unit or per spatial unit. PSD units are input-unit
        squared per cycle-per-coordinate-unit.
    window : str, tuple, array_like or None, default "hann"
        SciPy window specification. None uses a rectangular window.
    one_sided : bool, default True
        Return a one-sided density for real input. Interior positive-frequency
        power is doubled by SciPy; DC and Nyquist are not.
    detrend : str, callable or False, default False
        Forwarded to scipy.signal.periodogram. False preserves DC.

    Returns
    -------
    f, Pxx : ndarray
        Frequency coordinates and PSD density. For a uniformly sampled record,
        ``Pxx.sum() * (fs / len(x))`` estimates its mean square.

    Notes
    -----
    This uses window-power normalization (``scaling='density'``), not the
    coherent-gain amplitude normalization used by :func:`fft_spectrum`.
    """
    x = as_finite_1d_array(x)
    require_positive_finite_scalar(fs, "fs")
    if one_sided and np.iscomplexobj(x):
        raise ValueError("one_sided=True requires real-valued x")

    scipy_window = "boxcar" if window is None else window
    return periodogram(
        x,
        fs=fs,
        window=scipy_window,
        detrend=detrend,
        return_onesided=one_sided,
        scaling="density",
    )


def power_spectrum(x, fs=1.0, window="hann", one_sided=True,
                   detrend=False):
    """Return frequency bins and power per FFT bin.

    Power per bin is the density from :func:`psd_spectrum` multiplied by the
    bin width ``fs / len(x)``. Its unit is input-unit squared and its sum is the
    corresponding mean-square estimate. It is not an amplitude spectrum.
    """
    x = as_finite_1d_array(x)
    f, Pxx = psd_spectrum(
        x,
        fs=fs,
        window=window,
        one_sided=one_sided,
        detrend=detrend,
    )
    return f, Pxx * (fs / len(x))


def averaged_psd(x, block_size, overlap=0.5,
                 return_spectrogram=False, **psd_kwargs):
    """Average blockwise PSD estimates using a floored integer sample hop.

    Parameters
    ----------
    x : array_like
        Non-empty, finite 1D signal.
    block_size : int
        Number of samples per complete block.
    overlap : float, default 0.5
        Requested fractional overlap. The sample hop is
        ``int(block_size * (1 - overlap))``, matching :func:`averaged_fft`.
    return_spectrogram : bool, default False
        Also return block centers and individual block PSD densities.
    **psd_kwargs
        Forwarded to :func:`psd_spectrum`, for example ``fs=1000``,
        ``window='hann'`` or ``detrend='constant'``.

    Returns
    -------
    f, avg_psd : ndarray
        Frequency coordinates and arithmetic mean of block PSD densities.
    block_centers, Pxx_blocks : ndarray, optional
        Block-center coordinates and PSD densities with shape
        ``(number_of_blocks, number_of_bins)``.

    Notes
    -----
    This averages power density, unlike :func:`averaged_fft`, which averages
    linear magnitudes. Incomplete trailing blocks are omitted.
    """
    x = as_finite_1d_array(x)
    require_positive_integer(block_size, "block_size")
    try:
        valid_overlap = bool(np.isfinite(overlap) and 0 <= overlap < 1)
    except (TypeError, ValueError):
        valid_overlap = False
    if not valid_overlap:
        raise ValueError("overlap must be finite and satisfy 0 <= overlap < 1")

    step = int(block_size * (1 - overlap))
    if step < 1:
        raise ValueError("overlap produces a sample hop smaller than one")
    if len(x) < block_size:
        raise ValueError("signal length must be at least block_size")

    starts = np.arange(0, len(x) - block_size + 1, step)
    spectra = []
    for start in starts:
        f, Pxx = psd_spectrum(x[start:start + block_size], **psd_kwargs)
        spectra.append(Pxx)

    Pxx_blocks = np.stack(spectra)
    avg_psd = Pxx_blocks.mean(axis=0)
    if return_spectrogram:
        fs = psd_kwargs.get("fs", 1.0)
        block_centers = (starts + (block_size - 1) / 2) / fs
        return f, avg_psd, block_centers, Pxx_blocks
    return f, avg_psd


def averaged_fft(x, block_size, overlap=0.5,
                 return_spectrogram=False, **fft_kwargs):
    """Average the magnitudes of windowed, overlapping FFT blocks.

    Parameters
    ----------
    x : array_like
        1D input signal.
    block_size : int
        Number of input samples per block.
    overlap : float, default 0.5
        Requested fractional overlap (0 <= overlap < 1). The integer sample
        hop is ``int(block_size * (1 - overlap))``, which floors a non-integer
        hop. The effective overlap can therefore be slightly larger than the
        requested value.
    return_spectrogram : bool, default False
        Also return block-center coordinates and per-block magnitudes.
    **fft_kwargs
        Forwarded to fft_spectrum(), for example fs=1000, window='hann'.

    Returns
    -------
    f, avg : ndarray
        Frequency bins and mean magnitude spectrum.
    block_centers, S : ndarray, optional
        Block-center coordinates and magnitude spectra (blocks x bins).
        Centers are in time units for temporal input or position units for
        spatial input.

    Example
    -------
    >>> sample_index = np.arange(1024)
    >>> x = np.sin(2 * np.pi * 16 * sample_index / 256)
    >>> f, avg = averaged_fft(x, 256, fs=256, overlap=0.5)
    >>> f, avg, block_centers, S = averaged_fft(
    ...     x, 256, fs=256, return_spectrogram=True)

    Notes
    -----
    This is a linear-amplitude average, not a Welch PSD.
    Incomplete trailing blocks are ignored.
    Effective overlap is ``1 - step / block_size`` using the floored integer
    ``step`` described above.
    """
    x = as_finite_1d_array(x)
    require_positive_integer(block_size, "block_size")
    try:
        valid_overlap = bool(np.isfinite(overlap) and 0 <= overlap < 1)
    except (TypeError, ValueError):
        valid_overlap = False
    if not valid_overlap:
        raise ValueError("overlap must be finite and satisfy 0 <= overlap < 1")

    step = int(block_size * (1 - overlap))
    if step < 1:
        raise ValueError("overlap produces a sample hop smaller than one")
    if len(x) < block_size:
        raise ValueError("signal length must be at least block_size")

    starts = np.arange(0, len(x) - block_size + 1, step)
    spectra = []
    for start in starts:
        f, Xf = fft_spectrum(x[start:start + block_size], **fft_kwargs)
        spectra.append(np.abs(Xf))

    S = np.stack(spectra)
    avg = S.mean(axis=0)

    if return_spectrogram:
        fs = fft_kwargs.get("fs", 1.0)
        block_centers = (starts + (block_size - 1) / 2) / fs
        return f, avg, block_centers, S
    return f, avg
