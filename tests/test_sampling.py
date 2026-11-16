import numpy as np
import pytest

from signal_analyzer.fft_tools import fft_spectrum
from signal_analyzer.sampling import align_to_reference, resample_uniform


def test_align_to_reference_interpolates_only_the_coordinate_overlap():
    reference_coordinate = np.arange(6.0)
    reference_x = reference_coordinate**2
    coordinate = np.array([1.0, 3.0, 5.0])
    x = 2 * coordinate + 1

    common, reference_aligned, x_aligned = align_to_reference(
        reference_x, reference_coordinate, x, coordinate
    )

    np.testing.assert_array_equal(common, np.arange(1.0, 6.0))
    np.testing.assert_array_equal(reference_aligned, common**2)
    np.testing.assert_allclose(x_aligned, 2 * common + 1)


def test_align_to_reference_supports_nonuniform_coordinates():
    reference_coordinate = np.array([0.0, 0.4, 1.1, 2.0, 3.2])
    reference_x = np.sin(reference_coordinate)
    coordinate = np.array([0.2, 0.8, 2.4, 3.0])
    x = 3 * coordinate - 2

    common, reference_aligned, x_aligned = align_to_reference(
        reference_x, reference_coordinate, x, coordinate
    )

    np.testing.assert_array_equal(common, [0.4, 1.1, 2.0])
    np.testing.assert_array_equal(
        reference_aligned, np.sin(reference_coordinate)[[1, 2, 3]]
    )
    np.testing.assert_allclose(x_aligned, 3 * common - 2)


def test_align_to_reference_includes_shared_overlap_endpoints():
    common, reference_aligned, x_aligned = align_to_reference(
        np.array([10.0, 11.0, 12.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([21.0, 22.0, 23.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_array_equal(common, [1.0, 2.0])
    np.testing.assert_array_equal(reference_aligned, [11.0, 12.0])
    np.testing.assert_array_equal(x_aligned, [21.0, 22.0])


def test_align_to_reference_supports_complex_values_and_preserves_inputs():
    reference_coordinate = np.array([0.0, 1.0, 2.0, 3.0])
    reference_x = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
    coordinate = np.array([0.5, 1.5, 2.5])
    x = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    originals = tuple(
        array.copy()
        for array in (reference_x, reference_coordinate, x, coordinate)
    )

    common, reference_aligned, x_aligned = align_to_reference(
        reference_x, reference_coordinate, x, coordinate
    )

    np.testing.assert_array_equal(common, [1.0, 2.0])
    np.testing.assert_array_equal(reference_aligned, [2 + 2j, 3 + 3j])
    np.testing.assert_allclose(x_aligned, [2 + 3j, 4 + 5j])
    for actual, expected in zip(
        (reference_x, reference_coordinate, x, coordinate), originals
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("reference_coordinate", "coordinate"),
    [
        (np.array([0.0, 1.0]), np.array([2.0, 3.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0])),
    ],
)
def test_align_to_reference_rejects_insufficient_overlap(
    reference_coordinate, coordinate
):
    with pytest.raises(ValueError, match="at least two reference samples"):
        align_to_reference(
            np.ones(reference_coordinate.size),
            reference_coordinate,
            np.ones(coordinate.size),
            coordinate,
        )


@pytest.mark.parametrize(
    ("reference_x", "reference_coordinate", "x", "coordinate", "message"),
    [
        ([1.0], [0.0], [1.0, 2.0], [0.0, 1.0], "reference_x"),
        ([1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0], [0.0, 1.0], "equal length"),
        ([1.0, 2.0], [0.0, 0.0], [1.0, 2.0], [0.0, 1.0], "strictly increasing"),
        ([1.0, np.nan], [0.0, 1.0], [1.0, 2.0], [0.0, 1.0], "finite"),
        ([1.0, 2.0], [0.0, 1.0], [1.0], [0.0], "x"),
        ([1.0, 2.0], [0.0, 1.0], [1.0, 2.0], [0.0, np.inf], "finite"),
    ],
)
def test_align_to_reference_rejects_invalid_inputs(
    reference_x, reference_coordinate, x, coordinate, message
):
    with pytest.raises(ValueError, match=message):
        align_to_reference(reference_x, reference_coordinate, x, coordinate)


def test_resample_uniform_preserves_low_frequency_amplitude():
    fs = 960.0
    t = np.arange(1920) / fs
    x = 1.7 * np.sin(2 * np.pi * 30 * t)

    y, fs_out = resample_uniform(x, fs, down=4)
    f, Xf = fft_spectrum(y, fs=fs_out, window=None)

    assert fs_out == 240.0
    assert len(y) == 480
    assert abs(Xf[np.argmin(abs(f - 30))]) == pytest.approx(1.7, rel=2e-3)


def test_resample_uniform_suppresses_alias_above_output_nyquist():
    fs = 960.0
    t = np.arange(3840) / fs
    x = np.sin(2 * np.pi * 180 * t)

    y, fs_out = resample_uniform(x, fs, down=4)
    f, Xf = fft_spectrum(y, fs=fs_out, window=None)

    alias_bin = np.argmin(abs(f - 60))
    assert abs(Xf[alias_bin]) < 2e-3
    assert np.sqrt(np.mean(y[32:-32] ** 2)) < 2e-3


def test_resample_uniform_supports_rational_factors_and_reports_rate():
    x = np.linspace(0, 1, 100)

    y, fs_out = resample_uniform(x, fs=300.0, up=2, down=3)

    assert len(y) == 67
    assert fs_out == 200.0


def test_resample_uniform_respects_selected_axis():
    x = np.vstack([np.arange(100), np.arange(100)])

    y, fs_out = resample_uniform(x, fs=100.0, up=2, down=5, axis=1)

    assert y.shape == (2, 40)
    assert fs_out == 40.0


def test_resample_uniform_handles_short_valid_input():
    y, fs_out = resample_uniform(np.array([1.0, 2.0]), fs=10.0, down=3)

    assert y.shape == (1,)
    assert fs_out == pytest.approx(10 / 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fs": 0.0}, "fs must be a positive finite scalar"),
        ({"fs": 10.0, "up": 0}, "up must be a positive integer"),
        ({"fs": 10.0, "down": 1.5}, "down must be a positive integer"),
        ({"fs": 10.0, "axis": True}, "axis must be an integer"),
        ({"fs": 10.0, "axis": 2}, "axis is out of range"),
    ],
)
def test_resample_uniform_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resample_uniform(np.ones(8), **kwargs)


@pytest.mark.parametrize(
    "x",
    [
        np.array(1.0),
        np.ones((2, 1)),
        np.array([1.0, np.nan]),
    ],
)
def test_resample_uniform_rejects_invalid_data(x):
    with pytest.raises(ValueError):
        resample_uniform(x, fs=10.0)
