import numpy as np
import pytest

from signal_analyzer.demodulation import amplitude_demodulate
from signal_analyzer.resampling import speed_resample_aa


def _tone_amplitude(x, t, frequency):
    basis = np.exp(-2j * np.pi * frequency * t)
    return 2 * np.abs(np.dot(x, basis)) / len(x)


def _am_signal(fs=4096.0, duration=4.0):
    t = np.arange(int(fs * duration)) / fs
    carrier_amplitude = 2.0
    modulation_depth = 0.3
    modulation_frequency = 17.0
    envelope = carrier_amplitude * (
        1 + modulation_depth * np.cos(2 * np.pi * modulation_frequency * t)
    )
    x = envelope * np.cos(2 * np.pi * 500.0 * t)
    return t, x, carrier_amplitude, modulation_depth, modulation_frequency


def test_hilbert_demodulation_recovers_amplitude_modulation():
    fs = 4096.0
    t, x, amplitude, depth, modulation_frequency = _am_signal(fs=fs)

    y = amplitude_demodulate(
        x, fs, band=(400.0, 600.0), lowpass=60.0, detector="hilbert"
    )

    trim = int(fs)
    measured = _tone_amplitude(
        y[trim:-trim], t[trim:-trim], modulation_frequency
    )
    assert y.shape == x.shape
    assert measured == pytest.approx(amplitude * depth, rel=1e-2)


def test_square_demodulation_has_explicit_squared_units():
    fs = 4096.0
    t, x, amplitude, depth, modulation_frequency = _am_signal(fs=fs)

    y = amplitude_demodulate(
        x, fs, band=(400.0, 600.0), lowpass=60.0, detector="square"
    )

    trim = int(fs)
    measured = _tone_amplitude(
        y[trim:-trim], t[trim:-trim], modulation_frequency
    )
    assert measured == pytest.approx(amplitude**2 * depth, rel=1e-2)


def test_demodulation_can_retain_or_remove_envelope_mean():
    fs = 4096.0
    _, x, amplitude, _, _ = _am_signal(fs=fs)

    with_mean = amplitude_demodulate(
        x, fs, band=(400.0, 600.0), lowpass=60.0,
        detector="hilbert", remove_mean=False,
    )
    without_mean = amplitude_demodulate(
        x, fs, band=(400.0, 600.0), lowpass=60.0,
        detector="hilbert", remove_mean=True,
    )

    trim = int(fs)
    assert np.mean(with_mean[trim:-trim]) == pytest.approx(amplitude, rel=1e-2)
    assert abs(np.mean(without_mean)) < 1e-12


def test_demodulation_composes_before_spatial_resampling():
    fs = 4096.0
    duration = 4.0
    speed_value = 8.0
    spatial_order = 3.0
    modulation_depth = 0.4
    t = np.arange(int(fs * duration)) / fs
    speed = np.full(t.shape, speed_value)
    modulation_frequency = spatial_order * speed_value
    x = (
        1 + modulation_depth * np.cos(2 * np.pi * modulation_frequency * t)
    ) * np.cos(2 * np.pi * 600.0 * t)

    demodulated = amplitude_demodulate(
        x, fs, band=(500.0, 700.0), lowpass=80.0
    )
    trim = int(0.5 * fs)
    pos, y = speed_resample_aa(
        demodulated[trim:-trim],
        speed[trim:-trim],
        fs,
        resolution=64,
    )

    measured = 2 * np.abs(
        np.dot(y, np.exp(-2j * np.pi * spatial_order * pos))
    ) / len(y)
    assert measured == pytest.approx(modulation_depth, rel=2e-2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"band": (600.0, 400.0)}, "band must satisfy"),
        ({"band": (400.0,)}, "exactly two"),
        ({"band": (400.0, 600.0), "lowpass": 0.0}, "positive finite"),
        ({"band": (400.0, 600.0), "lowpass": 2500.0}, "below fs / 2"),
        ({"band": (400.0, 600.0), "detector": "rectify"}, "detector"),
        ({"band": (400.0, 600.0), "filter_order": 0}, "positive integer"),
    ],
)
def test_demodulation_rejects_invalid_filter_parameters(kwargs, message):
    parameters = {"band": (400.0, 600.0), "lowpass": 80.0}
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        amplitude_demodulate(np.ones(64), fs=4096.0, **parameters)


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (np.ones((2, 2)), "1D array"),
        (np.ones(1), "at least two"),
        (np.array([1.0, np.nan]), "finite"),
        (np.ones(8, dtype=complex), "real-valued"),
    ],
)
def test_demodulation_rejects_invalid_signal(x, message):
    with pytest.raises(ValueError, match=message):
        amplitude_demodulate(
            x, fs=4096.0, band=(400.0, 600.0), lowpass=80.0
        )
