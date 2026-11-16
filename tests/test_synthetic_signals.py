import numpy as np
import pytest

from signal_analyzer.demodulation import amplitude_demodulate
from signal_analyzer.fft_tools import fft_spectrum
from signal_analyzer.resampling import speed_resample_aa
from signal_analyzer.synthetic_signals import (
    simulate_bearing_fault,
    synthesize_tone_mix,
)


def test_tone_mix_recovers_temporal_and_order_components():
    fs = 1024.0
    t, speed, position, x = synthesize_tone_mix(
        fs=fs,
        duration=2.0,
        speed=8.0,
        temporal_components=((50.0, 1.2, 0.0),),
        order_components=((3.0, 0.7, 0.0),),
    )

    f, Xf = fft_spectrum(x, fs=fs, window=None)
    np.testing.assert_allclose(speed, 8.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(position, 8.0 * t, rtol=0.0, atol=1e-12)
    assert abs(Xf[np.argmin(abs(f - 50.0))]) == pytest.approx(1.2, rel=1e-12)
    assert abs(Xf[np.argmin(abs(f - 24.0))]) == pytest.approx(0.7, rel=1e-12)


def test_tone_mix_callable_speed_has_analytic_trapezoidal_position():
    fs = 1000.0
    t, speed, position, x = synthesize_tone_mix(
        fs=fs,
        duration=2.0,
        speed=lambda time: 2.0 + 0.5 * time,
        order_components=((4.0, 1.0, 0.0),),
    )

    np.testing.assert_allclose(speed, 2.0 + 0.5 * t, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        position, 2.0 * t + 0.25 * t**2, rtol=0.0, atol=1e-12
    )
    pos, y = speed_resample_aa(x, speed, fs, resolution=64)
    measured = 2 * np.abs(np.dot(y, np.exp(-2j * np.pi * 4.0 * pos))) / len(y)
    assert measured == pytest.approx(1.0, rel=1e-2)


def test_tone_mix_noise_is_deterministic_for_seed():
    first = synthesize_tone_mix(noise_std=0.2, seed=17)
    second = synthesize_tone_mix(noise_std=0.2, seed=17)
    third = synthesize_tone_mix(noise_std=0.2, seed=18)

    np.testing.assert_array_equal(first[3], second[3])
    assert not np.array_equal(first[3], third[3])


def test_bearing_fault_has_expected_speed_position_and_is_deterministic():
    kwargs = dict(
        fs=2048.0,
        duration=2.0,
        speed_min=8.0,
        speed_max=16.0,
        speed_cycles=1.0,
        resonance_frequency=600.0,
        noise_std=0.02,
        seed=9,
    )
    first = simulate_bearing_fault(**kwargs)
    second = simulate_bearing_fault(**kwargs)
    t, speed, position, x = first

    assert t.shape == speed.shape == position.shape == x.shape
    assert speed.min() == pytest.approx(8.0)
    assert speed.max() == pytest.approx(16.0)
    expected_position = np.r_[
        0.0,
        np.cumsum((speed[:-1] + speed[1:]) / (2 * kwargs["fs"])),
    ]
    np.testing.assert_allclose(position, expected_position, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(x, second[3])


def test_bearing_fault_envelope_recovers_fault_order():
    fs = 4096.0
    fault_order = 4.8
    t, speed, _, x = simulate_bearing_fault(
        fs=fs,
        duration=6.0,
        speed_min=8.0,
        speed_max=16.0,
        speed_cycles=1.0,
        shaft_order_amplitudes=(0.3, 0.15, 0.08, 0.04, 0.02),
        constant_components=(),
        fault_order=fault_order,
        fault_amplitude=1.0,
        resonance_frequency=1000.0,
        resonance_decay=0.003,
        noise_std=0.01,
        seed=3,
    )
    envelope = amplitude_demodulate(
        x, fs, band=(800.0, 1200.0), lowpass=200.0, detector="hilbert"
    )
    trim = int(0.5 * fs)
    pos, y = speed_resample_aa(
        envelope[trim:-trim], speed[trim:-trim], fs, resolution=64
    )
    f, Xf = fft_spectrum(y, fs=64, window="hann")
    search = (f >= 3.0) & (f <= 7.0)
    measured_order = f[search][np.argmax(np.abs(Xf[search]))]

    assert measured_order == pytest.approx(fault_order, abs=64 / len(y))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fs": 0.0}, "positive finite"),
        ({"duration": 0.0}, "positive finite"),
        ({"speed": 0.0}, "finite positive"),
        ({"speed": np.ones(3)}, "matching 1D"),
        ({"temporal_components": ((6000.0, 1.0, 0.0),)}, "below fs / 2"),
        ({"order_components": ((0.0, 1.0, 0.0),)}, "positive coordinate"),
        ({"noise_std": -1.0}, "non-negative"),
    ],
)
def test_tone_mix_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        synthesize_tone_mix(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speed_min": 0.0}, "positive finite"),
        ({"speed_min": 20.0, "speed_max": 10.0}, "greater than"),
        ({"shaft_order_amplitudes": (1.0, 0.5)}, "five finite"),
        ({"resonance_frequency": 3000.0}, "below fs / 2"),
        ({"fault_order": 500.0}, "fault event frequency"),
        ({"noise_std": -0.1}, "non-negative"),
    ],
)
def test_bearing_fault_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_bearing_fault(**kwargs)
