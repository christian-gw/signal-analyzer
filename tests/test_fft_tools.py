import numpy as np
import pytest

from signal_analyzer.fft_tools import (
    averaged_fft,
    averaged_psd,
    fft_spectrum,
    power_spectrum,
    psd_spectrum,
)


@pytest.mark.parametrize("window", [None, "hann"])
def test_fft_spectrum_recovers_bin_centered_sinusoid_amplitude(window):
    fs = 1024.0
    n = 1024
    bin_index = 73
    amplitude = 2.5
    t = np.arange(n) / fs
    x = amplitude * np.sin(2 * np.pi * bin_index * t)

    f, Xf = fft_spectrum(x, fs=fs, window=window)

    np.testing.assert_allclose(f[bin_index], bin_index, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.abs(Xf[bin_index]), amplitude, rtol=1e-12, atol=1e-12
    )


def test_fft_spectrum_does_not_double_dc_or_even_nyquist():
    n = 256
    dc_amplitude = 1.25
    nyquist_amplitude = 0.75
    x = dc_amplitude + nyquist_amplitude * (-1.0) ** np.arange(n)

    f, Xf = fft_spectrum(x, fs=n, window=None)

    np.testing.assert_allclose(f[[0, -1]], [0.0, n / 2], rtol=0, atol=0)
    np.testing.assert_allclose(
        Xf[[0, -1]], [dc_amplitude, nyquist_amplitude],
        rtol=1e-12, atol=1e-12,
    )


def test_fft_spectrum_doubles_last_positive_bin_for_odd_length():
    n = 255
    bin_index = n // 2
    amplitude = 1.75
    sample_index = np.arange(n)
    x = amplitude * np.cos(2 * np.pi * bin_index * sample_index / n)

    f, Xf = fft_spectrum(x, fs=n, window=None)

    assert len(f) == n // 2 + 1
    np.testing.assert_allclose(f[-1], bin_index, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.abs(Xf[-1]), amplitude, rtol=1e-12, atol=1e-12
    )


def test_two_sided_fft_preserves_amplitude_and_phase():
    n = 256
    bin_index = 19
    amplitude = 3.0
    phase = 0.37
    sample_index = np.arange(n)
    x = amplitude * np.cos(
        2 * np.pi * bin_index * sample_index / n + phase
    )

    f, Xf = fft_spectrum(x, fs=n, window=None, one_sided=False)

    expected_positive = (amplitude / 2) * np.exp(1j * phase)
    expected_negative = (amplitude / 2) * np.exp(-1j * phase)
    np.testing.assert_allclose(f[[bin_index, -bin_index]], [bin_index, -bin_index])
    np.testing.assert_allclose(
        Xf[[bin_index, -bin_index]],
        [expected_positive, expected_negative],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("overlap, expected_blocks", [(0.0, 4), (0.5, 7)])
def test_averaged_fft_of_repeated_signal_matches_one_block(
    overlap, expected_blocks
):
    block_size = 256
    sample_index = np.arange(block_size)
    block = 1.8 * np.sin(2 * np.pi * 23 * sample_index / block_size)
    x = np.tile(block, 4)

    f_expected, Xf_expected = fft_spectrum(
        block, fs=block_size, window="hann"
    )
    f, avg, _, S = averaged_fft(
        x,
        block_size,
        overlap=overlap,
        fs=block_size,
        window="hann",
        return_spectrogram=True,
    )

    assert S.shape == (expected_blocks, len(f_expected))
    np.testing.assert_allclose(f, f_expected, rtol=0, atol=0)
    np.testing.assert_allclose(avg, np.abs(Xf_expected), rtol=1e-12, atol=1e-12)


def test_averaged_fft_ignores_incomplete_trailing_block():
    block_size = 128
    sample_index = np.arange(block_size)
    block = np.sin(2 * np.pi * 11 * sample_index / block_size)
    x = np.concatenate([block, block, np.full(31, 1e6)])

    f, avg, _, S = averaged_fft(
        x,
        block_size,
        overlap=0,
        fs=block_size,
        window=None,
        return_spectrogram=True,
    )
    f_expected, Xf_expected = fft_spectrum(block, fs=block_size, window=None)

    assert S.shape == (2, len(f_expected))
    np.testing.assert_allclose(f, f_expected, rtol=0, atol=0)
    np.testing.assert_allclose(avg, np.abs(Xf_expected), rtol=1e-12, atol=1e-12)


def test_spectrogram_matches_average_centers_and_preserves_input():
    fs = 200.0
    block_size = 100
    overlap = 0.5
    sample_index = np.arange(260)
    x = (
        np.sin(2 * np.pi * 12 * sample_index / fs)
        + 0.2 * np.cos(2 * np.pi * 31 * sample_index / fs)
    )
    x_before = x.copy()

    f_plain, avg_plain = averaged_fft(
        x, block_size, overlap=overlap, fs=fs, window="hann"
    )
    f, avg, centers, S = averaged_fft(
        x,
        block_size,
        overlap=overlap,
        fs=fs,
        window="hann",
        return_spectrogram=True,
    )

    starts = np.arange(0, len(x) - block_size + 1, block_size // 2)
    expected_centers = (starts + (block_size - 1) / 2) / fs
    assert S.shape == (len(starts), len(f))
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_allclose(f, f_plain, rtol=0, atol=0)
    np.testing.assert_allclose(avg, avg_plain, rtol=0, atol=0)
    np.testing.assert_allclose(avg, S.mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(centers, expected_centers, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("block_size", "overlap", "signal_length", "expected_starts"),
    [
        (5, 0.5, 13, np.array([0, 2, 4, 6, 8])),
        (3, 0.5, 7, np.array([0, 1, 2, 3, 4])),
    ],
)
def test_averaged_fft_floors_non_integer_hop(
    block_size, overlap, signal_length, expected_starts
):
    fs = 10.0
    x = np.arange(signal_length, dtype=float)

    _, avg, centers, S = averaged_fft(
        x,
        block_size,
        overlap=overlap,
        fs=fs,
        window=None,
        return_spectrogram=True,
    )

    step = int(block_size * (1 - overlap))
    expected_centers = (
        expected_starts + (block_size - 1) / 2
    ) / fs
    assert step < block_size * (1 - overlap)
    assert S.shape[0] == len(expected_starts)
    np.testing.assert_allclose(centers, expected_centers, rtol=0, atol=0)
    np.testing.assert_allclose(avg, S.mean(axis=0), rtol=0, atol=0)


@pytest.mark.parametrize("x", [np.array([]), np.ones((2, 2))])
def test_fft_spectrum_rejects_empty_or_non_1d_input(x):
    with pytest.raises(ValueError, match="non-empty 1D"):
        fft_spectrum(x)


@pytest.mark.parametrize("fs", [0.0, -1.0, np.nan, np.inf])
def test_fft_spectrum_rejects_invalid_sampling_rate(fs):
    with pytest.raises(ValueError, match="fs must be a positive finite scalar"):
        fft_spectrum(np.ones(8), fs=fs)


@pytest.mark.parametrize(
    "x",
    [
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, np.inf, 1.0]),
    ],
)
def test_fft_spectrum_rejects_non_finite_signal(x):
    with pytest.raises(ValueError, match="x must contain only finite values"):
        fft_spectrum(x)


def test_fft_spectrum_requires_real_input_for_one_sided_output():
    x = np.ones(8, dtype=complex)

    with pytest.raises(ValueError, match="one_sided=True requires real-valued x"):
        fft_spectrum(x)

    f, Xf = fft_spectrum(x, one_sided=False)
    assert f.shape == Xf.shape == x.shape


@pytest.mark.parametrize("block_size", [0, -1, 2.5, True])
def test_averaged_fft_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        averaged_fft(np.ones(8), block_size)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, np.nan, np.inf])
def test_averaged_fft_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap must be finite"):
        averaged_fft(np.ones(8), block_size=4, overlap=overlap)


@pytest.mark.parametrize("one_sided", [True, False])
def test_psd_spectrum_integrates_to_mean_square(one_sided):
    fs = 800.0
    rng = np.random.default_rng(21)
    x = rng.normal(size=2048) + 0.25

    f, Pxx = psd_spectrum(
        x, fs=fs, window=None, one_sided=one_sided, detrend=False
    )

    assert f.shape == Pxx.shape
    np.testing.assert_allclose(
        Pxx.sum() * fs / len(x), np.mean(x**2), rtol=2e-15, atol=2e-15
    )


@pytest.mark.parametrize("window", [None, "hann"])
def test_psd_spectrum_known_sinusoid_has_expected_total_power(window):
    fs = 1024.0
    n = 2048
    amplitude = 3.0
    t = np.arange(n) / fs
    x = amplitude * np.sin(2 * np.pi * 73 * t)

    _, Pxx = psd_spectrum(x, fs=fs, window=window)

    np.testing.assert_allclose(
        Pxx.sum() * fs / n, amplitude**2 / 2, rtol=2e-12, atol=2e-12
    )


def test_power_spectrum_preserves_dc_and_nyquist_power():
    fs = 256.0
    n = 256
    dc = 1.25
    nyquist = 0.75
    x = dc + nyquist * (-1.0) ** np.arange(n)

    f, power = power_spectrum(x, fs=fs, window=None)

    np.testing.assert_allclose(f[[0, -1]], [0.0, fs / 2], rtol=0, atol=0)
    np.testing.assert_allclose(
        power[[0, -1]], [dc**2, nyquist**2], rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(power.sum(), np.mean(x**2), rtol=1e-13)


def test_power_spectrum_is_psd_times_bin_width():
    fs = 500.0
    x = np.random.default_rng(8).normal(size=1000)

    f_psd, Pxx = psd_spectrum(x, fs=fs, window="hann")
    f_power, power = power_spectrum(x, fs=fs, window="hann")

    np.testing.assert_array_equal(f_power, f_psd)
    np.testing.assert_allclose(
        power, Pxx * fs / len(x), rtol=1e-15, atol=0
    )


def test_psd_detrending_is_explicit_and_disabled_by_default():
    amplitude = 2.5
    x = np.full(512, amplitude)

    _, preserved = power_spectrum(x, fs=512, window=None)
    _, removed = power_spectrum(
        x, fs=512, window=None, detrend="constant"
    )

    np.testing.assert_allclose(preserved.sum(), amplitude**2, rtol=1e-14)
    np.testing.assert_allclose(removed, 0.0, rtol=0, atol=1e-28)


def test_hann_psd_of_seeded_noise_estimates_time_domain_power():
    rng = np.random.default_rng(123)
    x = rng.normal(scale=1.7, size=65536)

    _, Pxx = psd_spectrum(x, fs=4096, window="hann")
    spectral_power = Pxx.sum() * 4096 / len(x)

    assert spectral_power == pytest.approx(np.mean(x**2), rel=1.5e-2)


@pytest.mark.parametrize("overlap, expected_blocks", [(0.0, 4), (0.5, 7)])
def test_averaged_psd_matches_individual_repeated_block(
    overlap, expected_blocks
):
    block_size = 256
    sample_index = np.arange(block_size)
    block = 1.4 * np.sin(2 * np.pi * 29 * sample_index / block_size)
    x = np.tile(block, 4)
    f_expected, Pxx_expected = psd_spectrum(
        block, fs=block_size, window="hann"
    )

    f, avg_psd, centers, Pxx_blocks = averaged_psd(
        x,
        block_size,
        overlap=overlap,
        fs=block_size,
        window="hann",
        return_spectrogram=True,
    )

    assert Pxx_blocks.shape == (expected_blocks, len(f_expected))
    np.testing.assert_array_equal(f, f_expected)
    np.testing.assert_allclose(avg_psd, Pxx_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(avg_psd, Pxx_blocks.mean(axis=0), rtol=0, atol=0)
    step = int(block_size * (1 - overlap))
    starts = np.arange(0, len(x) - block_size + 1, step)
    expected_centers = (starts + (block_size - 1) / 2) / block_size
    np.testing.assert_allclose(centers, expected_centers, rtol=0, atol=0)


def test_averaged_psd_omits_incomplete_trailing_block():
    block_size = 128
    block = np.random.default_rng(4).normal(size=block_size)
    x = np.concatenate([block, block, np.full(37, 1e6)])

    _, avg_psd, _, Pxx_blocks = averaged_psd(
        x,
        block_size,
        overlap=0,
        fs=block_size,
        window=None,
        return_spectrogram=True,
    )
    _, expected = psd_spectrum(block, fs=block_size, window=None)

    assert Pxx_blocks.shape[0] == 2
    np.testing.assert_allclose(avg_psd, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("function", [psd_spectrum, power_spectrum])
def test_power_spectral_functions_reject_invalid_inputs(function):
    with pytest.raises(ValueError, match="non-empty 1D"):
        function(np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive finite scalar"):
        function(np.ones(8), fs=np.nan)
    with pytest.raises(ValueError, match="real-valued"):
        function(np.ones(8, dtype=complex), one_sided=True)


@pytest.mark.parametrize("block_size", [0, -1, 2.5, True])
def test_averaged_psd_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        averaged_psd(np.ones(8), block_size)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, np.nan, np.inf])
def test_averaged_psd_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap must be finite"):
        averaged_psd(np.ones(8), block_size=4, overlap=overlap)
