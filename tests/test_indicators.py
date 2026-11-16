import numpy as np
import pytest

from signal_analyzer.indicators import (
    amplitude_band_statistics,
    coordinate_range_mask,
    psd_band_statistics,
    signal_statistics,
)


def test_coordinate_range_mask_is_half_open():
    coordinate = np.arange(5.0)

    mask = coordinate_range_mask(coordinate, lower=1.0, upper=3.0)

    np.testing.assert_array_equal(mask, [False, True, True, False, False])


def test_coordinate_range_mask_supports_open_bounds_and_empty_selection():
    coordinate = np.arange(4.0)

    np.testing.assert_array_equal(
        coordinate_range_mask(coordinate, upper=2.0),
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        coordinate_range_mask(coordinate, lower=3.0),
        [False, False, False, True],
    )
    assert not np.any(coordinate_range_mask(coordinate, lower=10.0))


@pytest.mark.parametrize(
    ("coordinate", "kwargs", "message"),
    [
        (np.ones((2, 2)), {}, "1D array"),
        (np.array([0.0, np.nan]), {}, "finite"),
        (np.array([0.0, 1.0, 1.0]), {}, "strictly increasing"),
        (np.arange(3.0), {"lower": 2.0, "upper": 1.0}, "smaller"),
        (np.arange(3.0), {"lower": np.inf}, "finite scalar"),
    ],
)
def test_coordinate_range_mask_rejects_invalid_input(
    coordinate, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        coordinate_range_mask(coordinate, **kwargs)


def test_signal_statistics_known_population_values_and_energy():
    x = np.array([-1.0, 1.0, -1.0, 1.0])

    result = signal_statistics(x, sample_spacing=0.25)

    assert result == pytest.approx(
        {
            "count": 4,
            "mean": 0.0,
            "standard_deviation": 1.0,
            "mean_square": 1.0,
            "rms": 1.0,
            "absolute_peak": 1.0,
            "peak_to_peak": 2.0,
            "crest_factor": 1.0,
            "skewness": 0.0,
            "kurtosis": 1.0,
            "energy": 1.0,
        }
    )


def test_signal_statistics_omits_energy_without_spacing():
    result = signal_statistics(np.array([1.0, 2.0, 3.0]))

    assert "energy" not in result


@pytest.mark.parametrize("x", [np.zeros(4), np.full(4, 2.0)])
def test_signal_statistics_defines_constant_signal_ratios(x):
    result = signal_statistics(x)

    assert np.isnan(result["skewness"])
    assert np.isnan(result["kurtosis"])
    if np.all(x == 0):
        assert np.isnan(result["crest_factor"])
    else:
        assert result["crest_factor"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "sample_spacing", "message"),
    [
        (np.array([]), None, "non-empty"),
        (np.array([1.0, np.nan]), None, "finite"),
        (np.ones(2, dtype=complex), None, "real-valued"),
        (np.ones(2), 0.0, "positive"),
    ],
)
def test_signal_statistics_rejects_invalid_input(x, sample_spacing, message):
    with pytest.raises(ValueError, match=message):
        signal_statistics(x, sample_spacing=sample_spacing)


def test_amplitude_band_statistics_uses_magnitude_semantics():
    coordinate = np.arange(5.0)
    amplitude = np.array([0.0, 1.0, 3.0, 2.0, 10.0])

    result = amplitude_band_statistics(
        coordinate, amplitude, lower=1.0, upper=4.0
    )

    assert result == pytest.approx(
        {
            "count": 3,
            "peak_coordinate": 2.0,
            "peak_amplitude": 3.0,
            "mean_amplitude": 2.0,
            "median_amplitude": 2.0,
            "magnitude_integral": 4.5,
        }
    )
    assert "rms" not in result
    assert "power" not in result


def test_amplitude_band_statistics_single_bin_integral_is_zero():
    result = amplitude_band_statistics(
        np.array([2.0]), np.array([3.0])
    )

    assert result["magnitude_integral"] == 0.0


@pytest.mark.parametrize(
    ("coordinate", "amplitude", "kwargs", "message"),
    [
        (np.arange(3.0), np.ones(2), {}, "equal length"),
        (np.arange(3.0), np.array([1.0, -1.0, 1.0]), {}, "non-negative"),
        (np.arange(3.0), np.ones(3), {"lower": 5.0}, "selects no"),
    ],
)
def test_amplitude_band_statistics_rejects_invalid_input(
    coordinate, amplitude, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        amplitude_band_statistics(coordinate, amplitude, **kwargs)


def test_psd_band_statistics_calculates_power_rms_and_centers():
    frequency = np.arange(5.0)
    psd = np.array([0.0, 1.0, 4.0, 1.0, 0.0])

    result = psd_band_statistics(
        frequency, psd, lower=1.0, upper=4.0
    )

    assert result == pytest.approx(
        {
            "count": 3,
            "bin_width": 1.0,
            "band_power": 6.0,
            "band_rms": np.sqrt(6.0),
            "dominant_coordinate": 2.0,
            "power_weighted_center": 2.0,
        }
    )


def test_psd_band_statistics_zero_power_has_nan_coordinates():
    result = psd_band_statistics(
        np.arange(4.0), np.zeros(4), lower=1.0, upper=3.0
    )

    assert result["band_power"] == 0.0
    assert result["band_rms"] == 0.0
    assert np.isnan(result["dominant_coordinate"])
    assert np.isnan(result["power_weighted_center"])


@pytest.mark.parametrize(
    ("frequency", "psd", "message"),
    [
        (np.array([0.0, 1.0, 2.1]), np.ones(3), "uniformly spaced"),
        (np.arange(3.0), np.array([1.0, -0.1, 1.0]), "non-negative"),
        (np.arange(3.0), np.ones(2), "equal length"),
        (np.array([0.0]), np.ones(1), "at least"),
    ],
)
def test_psd_band_statistics_rejects_invalid_input(frequency, psd, message):
    with pytest.raises(ValueError, match=message):
        psd_band_statistics(frequency, psd)
