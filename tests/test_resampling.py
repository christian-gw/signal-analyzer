import numpy as np
import pytest

from signal_analyzer.resampling import (
    max_speed_aa_filter,
    speed_resample,
    speed_resample_aa,
)


def test_speed_resample_constant_speed_coordinates_and_linear_signal():
    fs = 100.0
    resolution = 10
    t = np.arange(100) / fs
    speed = np.full(t.shape, 2.0)
    sample_position = 2.0 * t
    x = 3.0 * sample_position - 1.0

    pos, y = speed_resample(x, speed, fs, resolution=resolution)

    expected_pos = np.arange(0.0, sample_position[-1], 1 / resolution)
    np.testing.assert_allclose(pos, expected_pos, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.diff(pos), 1 / resolution,
                               rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(y, 3.0 * pos - 1.0,
                               rtol=0.0, atol=1e-13)


def test_speed_resample_excludes_exact_final_grid_boundary():
    fs = 100.0
    resolution = 20
    x = np.arange(101, dtype=float)
    speed = np.full(x.shape, 2.0)

    pos, _ = speed_resample(x, speed, fs, resolution=resolution)

    # Repeated addition makes the integrated endpoint slightly greater than
    # 2.0 in binary floating point. The conceptual half-open grid still ends
    # at 1.95 rather than acquiring a spurious sample at 2.0.
    assert len(pos) == 40
    np.testing.assert_allclose(pos[-1], 1.95, rtol=0.0, atol=1e-14)
    assert np.all(pos < 2.0)


def test_speed_resample_includes_last_grid_point_below_non_grid_endpoint():
    x = np.arange(4, dtype=float)
    speed = np.ones(4)

    pos, _ = speed_resample(x, speed, fs=10.0, resolution=4)

    np.testing.assert_array_equal(pos, np.array([0.0, 0.25]))


def test_speed_resample_keeps_grid_point_below_genuinely_later_endpoint():
    x = np.array([0.0, 1.0])
    speed = np.full(2, 1.0001)

    pos, _ = speed_resample(x, speed, fs=1.0, resolution=10)

    assert len(pos) == 11
    np.testing.assert_allclose(pos[-1], 1.0, rtol=0.0, atol=0.0)
    assert pos[-1] < 1.0001


def test_speed_resample_linear_speed_matches_analytic_position_signal():
    fs = 1000.0
    resolution = 64
    t = np.arange(2001) / fs
    speed = 2.0 + 0.5 * t
    sample_position = 2.0 * t + 0.25 * t**2
    spatial_order = 3.0
    x = np.sin(2 * np.pi * spatial_order * sample_position)

    pos, y = speed_resample(x, speed, fs, resolution=resolution)

    expected_pos = np.arange(0.0, sample_position[-1], 1 / resolution)
    np.testing.assert_allclose(pos, expected_pos, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        y,
        np.sin(2 * np.pi * spatial_order * pos),
        rtol=0.0,
        atol=5e-4,
    )


def test_known_spatial_order_survives_positive_speed_run_up_and_down():
    fs = 1000.0
    duration = 2.0
    resolution = 128
    spatial_order = 5.0
    t = np.arange(int(duration * fs) + 1) / fs
    speed = 4.0 + 1.5 * np.sin(2 * np.pi * t / duration)
    sample_position = np.r_[
        0.0,
        np.cumsum((speed[:-1] + speed[1:]) / (2 * fs)),
    ]
    x = np.sin(2 * np.pi * spatial_order * sample_position)

    pos, y = speed_resample(x, speed, fs, resolution=resolution)
    frequencies = np.fft.rfftfreq(len(y), d=1 / resolution)
    amplitudes = 2 * np.abs(np.fft.rfft(y)) / len(y)
    peak = np.argmax(amplitudes[1:]) + 1

    frequency_spacing = resolution / len(y)
    assert frequencies[peak] == pytest.approx(
        spatial_order, abs=frequency_spacing / 2
    )
    assert amplitudes[peak] == pytest.approx(1.0, rel=6e-3)
    np.testing.assert_allclose(
        y,
        np.sin(2 * np.pi * spatial_order * pos),
        rtol=0.0,
        atol=4e-3,
    )
    np.testing.assert_allclose(np.diff(pos), 1 / resolution,
                               rtol=0.0, atol=1e-13)


def test_aa_wrapper_matches_target_grid_and_output_length():
    fs = 100.0
    resolution = 20
    t = np.arange(101) / fs
    speed = np.full(t.shape, 2.0)
    x = np.sin(2 * np.pi * 2.0 * speed * t)

    direct_pos, _ = speed_resample(x, speed, fs, resolution=resolution)
    aa_pos, aa_y = speed_resample_aa(
        x, speed, fs, resolution=resolution, oversample=4
    )

    np.testing.assert_allclose(aa_pos, direct_pos, rtol=0.0, atol=1e-14)
    assert len(aa_y) == len(aa_pos)
    np.testing.assert_allclose(np.diff(aa_pos), 1 / resolution,
                               rtol=0.0, atol=1e-14)


def test_resamplers_handle_a_short_valid_input_consistently():
    x = np.array([0.0, 1.0])
    speed = np.array([1.0, 1.0])

    direct_pos, direct_y = speed_resample(
        x, speed, fs=10.0, resolution=20
    )
    aa_pos, aa_y = speed_resample_aa(
        x, speed, fs=10.0, resolution=20, oversample=4
    )

    expected_pos = np.array([0.0, 0.05])
    np.testing.assert_allclose(direct_pos, expected_pos,
                               rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(aa_pos, expected_pos,
                               rtol=0.0, atol=1e-14)
    assert direct_y.shape == expected_pos.shape
    assert aa_y.shape == expected_pos.shape


@pytest.mark.parametrize("bad_speed", [
    np.array([1.0, 0.0, 1.0]),
    np.array([1.0, -0.1, 1.0]),
])
def test_speed_resample_rejects_nonpositive_speed(bad_speed):
    with pytest.raises(ValueError, match="positive"):
        speed_resample(np.ones(3), bad_speed, fs=10.0)


@pytest.mark.parametrize("fs", [0.0, -1.0, np.nan, np.inf])
def test_speed_resample_rejects_nonpositive_fs(fs):
    with pytest.raises(ValueError, match="fs must be a positive finite scalar"):
        speed_resample(np.ones(3), np.ones(3), fs=fs)


@pytest.mark.parametrize("resolution", [0.0, -1.0, np.nan, np.inf])
def test_speed_resample_rejects_nonpositive_resolution(resolution):
    with pytest.raises(
        ValueError, match="resolution must be a positive finite scalar"
    ):
        speed_resample(
            np.ones(3), np.ones(3), fs=10.0, resolution=resolution
        )


@pytest.mark.parametrize(
    ("x", "speed"),
    [
        (np.ones(3), np.ones(2)),
        (np.ones((2, 2)), np.ones(4)),
        (np.ones(4), np.ones((2, 2))),
    ],
)
def test_speed_resample_rejects_mismatched_or_non_1d_inputs(x, speed):
    with pytest.raises(ValueError, match="1D arrays of equal length"):
        speed_resample(x, speed, fs=10.0)


@pytest.mark.parametrize("oversample", [0, -1, 1.5, True, np.bool_(True)])
def test_aa_wrapper_rejects_invalid_oversample(oversample):
    with pytest.raises(ValueError, match="positive integer"):
        speed_resample_aa(
            np.ones(3), np.ones(3), fs=10.0, oversample=oversample
        )


def test_speed_resample_rejects_fewer_than_two_samples():
    with pytest.raises(ValueError, match="at least two samples"):
        speed_resample(np.ones(1), np.ones(1), fs=10.0)


@pytest.mark.parametrize(
    ("x", "speed", "message"),
    [
        (np.array([0.0, np.nan]), np.ones(2), "x must contain only finite"),
        (np.array([0.0, np.inf]), np.ones(2), "x must contain only finite"),
        (np.ones(2), np.array([1.0, np.nan]), "speed must contain only finite"),
        (np.ones(2), np.array([1.0, np.inf]), "speed must contain only finite"),
    ],
)
def test_speed_resample_rejects_non_finite_samples(x, speed, message):
    with pytest.raises(ValueError, match=message):
        speed_resample(x, speed, fs=10.0)


def _tone_amplitude(x, frequency, fs):
    sample_index = np.arange(len(x))
    basis = np.exp(-2j * np.pi * frequency * sample_index / fs)
    return 2 * np.abs(np.dot(x, basis)) / len(x)


def test_max_speed_aa_filter_preserves_below_cutoff_and_attenuates_above():
    fs = 1024.0
    n = 4096
    t = np.arange(n) / fs
    speed = np.full(n, 2.0)
    resolution = 128
    low_frequency = 64.0
    high_frequency = 256.0
    x = (
        np.sin(2 * np.pi * low_frequency * t)
        + np.sin(2 * np.pi * high_frequency * t)
    )

    y = max_speed_aa_filter(x, speed, fs, resolution=resolution)

    assert _tone_amplitude(y, low_frequency, fs) == pytest.approx(
        1.0, rel=2e-3
    )
    assert _tone_amplitude(y, high_frequency, fs) < 2e-3


def test_max_speed_aa_filter_uses_maximum_speed_cutoff():
    fs = 1024.0
    n = 4096
    t = np.arange(n) / fs
    speed = np.linspace(1.0, 4.0, n)
    resolution = 128
    frequency = 192.0
    x = np.sin(2 * np.pi * frequency * t)

    y = max_speed_aa_filter(x, speed, fs, resolution=resolution)

    # The 192 Hz tone is above the 64 Hz cutoff implied by minimum speed,
    # but below the 256 Hz cutoff implied by maximum speed.
    assert _tone_amplitude(y, frequency, fs) == pytest.approx(1.0, rel=3e-3)


def test_max_speed_aa_filter_skips_when_cutoff_reaches_input_nyquist():
    x = np.arange(32, dtype=float)
    speed = np.full(x.shape, 10.0)

    y = max_speed_aa_filter(x, speed, fs=100.0, resolution=20)

    np.testing.assert_array_equal(y, x)
    assert y is not x


@pytest.mark.parametrize("order", [0, -1, 2.5, True])
def test_max_speed_aa_filter_rejects_invalid_order(order):
    with pytest.raises(ValueError, match="order must be a positive integer"):
        max_speed_aa_filter(
            np.ones(16), np.ones(16), fs=100.0, resolution=20, order=order
        )


def test_aa_wrapper_applies_max_speed_temporal_filter():
    fs = 1024.0
    t = np.arange(4097) / fs
    speed = np.full(t.shape, 2.0)
    x = np.sin(2 * np.pi * 192 * t)

    _, y = speed_resample_aa(
        x, speed, fs, resolution=64, oversample=4
    )

    assert _tone_amplitude(y, 0.0, 64) < 1e-3
    assert np.sqrt(np.mean(y[32:-32] ** 2)) < 1e-3


def test_aa_wrapper_rejects_invalid_filter_order():
    with pytest.raises(ValueError, match="order must be a positive integer"):
        speed_resample_aa(
            np.ones(16), np.ones(16), fs=100.0, resolution=20,
            filter_order=0,
        )


def test_default_oversampling_covers_four_to_one_speed_ratio_working_band():
    fs = 4096.0
    duration = 2.0
    resolution = 64
    speed_min = 2.5
    speed_max = 10.0
    temporal_frequency = speed_max * resolution / 2.56
    t = np.arange(int(fs * duration) + 1) / fs
    speed = (
        (speed_min + speed_max) / 2
        + (speed_max - speed_min) / 2 * np.sin(2 * np.pi * t / duration)
    )
    x = np.sin(2 * np.pi * temporal_frequency * t)

    pos_default, y_default = speed_resample_aa(
        x, speed, fs, resolution=resolution, oversample=4
    )
    pos_dense, y_dense = speed_resample_aa(
        x, speed, fs, resolution=resolution, oversample=16
    )

    np.testing.assert_allclose(pos_default, pos_dense, rtol=0.0, atol=1e-14)
    edge = int(0.05 * len(y_dense))
    difference = y_default[edge:-edge] - y_dense[edge:-edge]
    relative_rms_error = (
        np.sqrt(np.mean(difference**2))
        / np.sqrt(np.mean(y_dense[edge:-edge] ** 2))
    )
    assert relative_rms_error < 5e-3
