"""Coordinate selection and scalar indicators for signals and spectra."""

import numpy as np

from ._validation import (
    as_finite_1d_array,
    require_positive_finite_scalar,
)

def _finite_bound(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar or None")
    try:
        finite = bool(np.isfinite(value))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite scalar or None")
    return float(value)


def _validated_coordinate(coordinate, name="coordinate", minimum_size=1):
    coordinate = as_finite_1d_array(
        coordinate, name, min_size=minimum_size, real=True
    )
    if coordinate.size > 1 and np.any(np.diff(coordinate) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return coordinate


def coordinate_range_mask(coordinate, lower=None, upper=None):
    """Return a mask for the half-open range ``lower <= coordinate < upper``.

    Either bound may be ``None``. An empty selection is represented by an
    all-false mask; indicator functions raise a clear error for such a range.
    """
    coordinate = _validated_coordinate(coordinate)
    lower = _finite_bound(lower, "lower")
    upper = _finite_bound(upper, "upper")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("lower must be smaller than upper")

    mask = np.ones(coordinate.shape, dtype=bool)
    if lower is not None:
        mask &= coordinate >= lower
    if upper is not None:
        mask &= coordinate < upper
    return mask


def signal_statistics(x, sample_spacing=None):
    """Return population statistics for a finite real-valued 1D signal.

    RMS and mean square are calculated directly from x. Crest factor is the
    absolute peak divided by RMS. Skewness and Pearson kurtosis (normal = 3)
    use population central moments; they are NaN for a constant signal.

    If ``sample_spacing`` is supplied, ``energy`` is the rectangular discrete
    integral ``sum(x**2) * sample_spacing``. Without spacing, no energy key is
    returned because the coordinate unit is unknown.
    """
    x = as_finite_1d_array(x, "x", real=True)
    x_float = x.astype(float, copy=False)
    mean = float(np.mean(x_float))
    centered = x_float - mean
    variance = float(np.mean(centered**2))
    standard_deviation = float(np.sqrt(variance))
    mean_square = float(np.mean(x_float**2))
    rms = float(np.sqrt(mean_square))
    absolute_peak = float(np.max(np.abs(x_float)))

    if standard_deviation == 0:
        skewness = np.nan
        kurtosis = np.nan
    else:
        normalized = centered / standard_deviation
        skewness = float(np.mean(normalized**3))
        kurtosis = float(np.mean(normalized**4))

    result = {
        "count": int(x.size),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "mean_square": mean_square,
        "rms": rms,
        "absolute_peak": absolute_peak,
        "peak_to_peak": float(np.ptp(x_float)),
        "crest_factor": absolute_peak / rms if rms > 0 else np.nan,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

    if sample_spacing is not None:
        require_positive_finite_scalar(sample_spacing, "sample_spacing")
        spacing = float(sample_spacing)
        result["energy"] = float(np.sum(x_float**2) * spacing)
    return result


def _selected_values(coordinate, values, lower, upper, value_name):
    coordinate = _validated_coordinate(coordinate)
    values = as_finite_1d_array(values, value_name, real=True)
    if values.shape != coordinate.shape:
        raise ValueError(f"{value_name} and coordinate must have equal length")
    mask = coordinate_range_mask(coordinate, lower=lower, upper=upper)
    if not np.any(mask):
        raise ValueError("coordinate range selects no samples")
    return coordinate[mask], values[mask]


def amplitude_band_statistics(
    coordinate, amplitude, lower=None, upper=None
):
    """Return magnitude-only statistics in a coordinate band.

    This function does not calculate power or RMS. ``magnitude_integral`` is a
    trapezoidal area under the amplitude curve and retains the corresponding
    amplitude-times-coordinate unit.
    """
    amplitude = as_finite_1d_array(amplitude, "amplitude", real=True)
    if np.any(amplitude < 0):
        raise ValueError("amplitude must be non-negative")
    selected_coordinate, selected_amplitude = _selected_values(
        coordinate, amplitude, lower, upper, "amplitude"
    )

    peak_index = int(np.argmax(selected_amplitude))
    return {
        "count": int(selected_amplitude.size),
        "peak_coordinate": float(selected_coordinate[peak_index]),
        "peak_amplitude": float(selected_amplitude[peak_index]),
        "mean_amplitude": float(np.mean(selected_amplitude)),
        "median_amplitude": float(np.median(selected_amplitude)),
        "magnitude_integral": float(
            np.trapz(selected_amplitude, selected_coordinate)
        ),
    }


def psd_band_statistics(frequency, psd, lower=None, upper=None):
    """Return power and RMS statistics from a uniformly spaced PSD.

    Band selection uses bin centers and the half-open range convention. Band
    power is ``sum(psd_in_band) * df`` and band RMS is its square root. No
    partial-bin weighting is applied. Dominant coordinate and power-weighted
    center are NaN when the selected PSD contains zero total power.
    """
    frequency = _validated_coordinate(
        frequency, name="frequency", minimum_size=2
    )
    psd = as_finite_1d_array(psd, "psd", min_size=2, real=True)
    if psd.shape != frequency.shape:
        raise ValueError("psd and frequency must have equal length")
    if np.any(psd < 0):
        raise ValueError("psd must be non-negative")

    differences = np.diff(frequency)
    df = float(differences[0])
    tolerance = 32 * np.finfo(float).eps * max(1.0, abs(df))
    if not np.allclose(differences, df, rtol=1e-10, atol=tolerance):
        raise ValueError("frequency must be uniformly spaced")

    mask = coordinate_range_mask(frequency, lower=lower, upper=upper)
    if not np.any(mask):
        raise ValueError("coordinate range selects no samples")
    selected_frequency = frequency[mask]
    selected_psd = psd[mask]
    psd_sum = float(np.sum(selected_psd))
    band_power = psd_sum * df

    if psd_sum == 0:
        dominant_coordinate = np.nan
        power_weighted_center = np.nan
    else:
        dominant_coordinate = float(
            selected_frequency[int(np.argmax(selected_psd))]
        )
        power_weighted_center = float(
            np.sum(selected_frequency * selected_psd) / psd_sum
        )

    return {
        "count": int(selected_psd.size),
        "bin_width": df,
        "band_power": band_power,
        "band_rms": float(np.sqrt(band_power)),
        "dominant_coordinate": dominant_coordinate,
        "power_weighted_center": power_weighted_center,
    }
