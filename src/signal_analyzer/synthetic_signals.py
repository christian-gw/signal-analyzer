"""Deterministic synthetic signals for generic DSP examples and tests."""

import numpy as np
from scipy.signal import lfilter

from ._validation import (
    require_nonnegative_finite_scalar,
    require_positive_finite_scalar,
)


def _time_grid(fs, duration):
    require_positive_finite_scalar(fs, "fs")
    require_positive_finite_scalar(duration, "duration")
    sample_count = int(np.round(fs * duration))
    if sample_count < 2:
        raise ValueError("fs * duration must produce at least two samples")
    return np.arange(sample_count, dtype=float) / fs


def _speed_array(speed, t):
    values = speed(t) if callable(speed) else speed
    if np.isscalar(values):
        values = np.full(t.shape, values, dtype=float)
    else:
        values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.shape != t.shape:
        raise ValueError("speed must be a scalar, a matching 1D array, or callable")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("speed must contain only finite positive values")
    return values


def _integrate_speed(speed, fs):
    return np.r_[
        0.0,
        np.cumsum((speed[:-1] + speed[1:]) / (2 * fs)),
    ]


def _component_tuples(components, name, maximum_frequency=None):
    checked = []
    try:
        iterator = iter(components)
    except TypeError:
        raise ValueError(f"{name} must be an iterable of three-value tuples") from None
    for component in iterator:
        try:
            coordinate, amplitude, phase = component
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} entries must be (coordinate, amplitude, phase)"
            ) from None
        try:
            valid = bool(
                np.isfinite(coordinate)
                and np.isfinite(amplitude)
                and np.isfinite(phase)
                and coordinate > 0
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} entries must contain finite values and a positive coordinate")
        if maximum_frequency is not None and coordinate >= maximum_frequency:
            raise ValueError(f"{name} frequencies must be below fs / 2")
        checked.append((float(coordinate), float(amplitude), float(phase)))
    return checked


def synthesize_tone_mix(
    fs=4096.0,
    duration=4.0,
    speed=10.0,
    temporal_components=(),
    order_components=(),
    noise_std=0.0,
    seed=0,
):
    """Synthesize additive temporal tones and speed-dependent order tones.

    Components are ``(coordinate, amplitude, phase)`` tuples. Temporal
    coordinates are cycles per time unit; order coordinates are cycles per
    integrated speed unit. Each component is a cosine. ``speed`` may be a
    positive scalar, a matching 1D array, or a callable evaluated as
    ``speed(t)``.

    Parameters
    ----------
    fs : float, default 4096
        Samples per time unit.
    duration : float, default 4
        Duration in time units; the grid is half-open at this duration.
    speed : scalar, array_like or callable, default 10
        Speed in integrated units per time unit.
    temporal_components, order_components : iterable
        ``(frequency_or_order, amplitude, phase_radians)`` tuples.
    noise_std : float, default 0
        Standard deviation of additive Gaussian noise, in signal units.
    seed : int or None, default 0
        Seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    t, speed, position, x : ndarray
        Synchronous 1D arrays. Position is trapezoidally integrated from speed.

    Example
    -------
    >>> t, speed, position, x = synthesize_tone_mix(
    ...     fs=1000, duration=1, speed=10,
    ...     temporal_components=((50, 1.0, 0.0),),
    ...     order_components=((3, 0.5, 0.0),))
    >>> len(t) == len(speed) == len(position) == len(x)
    True
    """
    t = _time_grid(fs, duration)
    speed_values = _speed_array(speed, t)
    position = _integrate_speed(speed_values, fs)
    temporal = _component_tuples(
        temporal_components, "temporal_components", maximum_frequency=fs / 2
    )
    orders = _component_tuples(order_components, "order_components")
    require_nonnegative_finite_scalar(noise_std, "noise_std")

    x = np.zeros(t.shape, dtype=float)
    for frequency, amplitude, phase in temporal:
        x += amplitude * np.cos(2 * np.pi * frequency * t + phase)
    for order, amplitude, phase in orders:
        x += amplitude * np.cos(2 * np.pi * order * position + phase)
    if noise_std:
        x += np.random.default_rng(seed).normal(0.0, noise_std, size=len(t))
    return t, speed_values, position, x


def simulate_bearing_fault(
    fs=4096.0,
    duration=8.0,
    speed_min=5.0,
    speed_max=20.0,
    speed_cycles=1.0,
    shaft_order_amplitudes=(1.0, 0.5, 0.3, 0.2, 0.1),
    constant_components=((60.0, 0.2, 0.0), (120.0, 0.1, 0.4)),
    fault_order=4.9,
    fault_amplitude=0.8,
    resonance_frequency=1000.0,
    resonance_decay=0.004,
    noise_std=0.05,
    seed=0,
):
    """Simulate a qualitative variable-speed motor and bearing-fault signal.

    Speed follows a smooth cosine run-up/run-down between ``speed_min`` and
    ``speed_max``. The signal combines shaft orders 1 through 5, fixed temporal
    tones and impacts at ``fault_order`` per integrated revolution. Each impact
    excites a normalized exponentially decaying sinusoid at
    ``resonance_frequency``.

    This is a generic workflow signal, not a calibrated mechanical model or
    acceleration prediction.

    Frequencies are cycles per time unit, speed is revolutions per time unit,
    position is revolutions, and ``resonance_decay`` is in time units.

    Returns
    -------
    t, speed, position, x : ndarray
        Synchronous time, speed, position and vibration arrays.

    Notes
    -----
    This is a qualitative public bearing-fault model for testing processing
    workflows, not a mechanical bearing model or calibrated acceleration signal.
    """
    t = _time_grid(fs, duration)
    require_positive_finite_scalar(speed_min, "speed_min")
    require_positive_finite_scalar(speed_max, "speed_max")
    if speed_max < speed_min:
        raise ValueError("speed_max must be greater than or equal to speed_min")
    require_positive_finite_scalar(speed_cycles, "speed_cycles")
    require_positive_finite_scalar(fault_order, "fault_order")
    require_nonnegative_finite_scalar(fault_amplitude, "fault_amplitude")
    require_positive_finite_scalar(resonance_frequency, "resonance_frequency")
    if resonance_frequency >= fs / 2:
        raise ValueError("resonance_frequency must be below fs / 2")
    require_positive_finite_scalar(resonance_decay, "resonance_decay")
    require_nonnegative_finite_scalar(noise_std, "noise_std")

    amplitudes = np.asarray(shaft_order_amplitudes, dtype=float)
    if amplitudes.shape != (5,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("shaft_order_amplitudes must contain five finite values")
    constant = _component_tuples(
        constant_components, "constant_components", maximum_frequency=fs / 2
    )
    if fault_order * speed_max >= fs / 2:
        raise ValueError("fault event frequency must remain below fs / 2")

    speed_mean = (speed_min + speed_max) / 2
    speed_amplitude = (speed_max - speed_min) / 2
    speed = speed_mean - speed_amplitude * np.cos(
        2 * np.pi * speed_cycles * t / duration
    )
    position = _integrate_speed(speed, fs)

    x = np.zeros(t.shape, dtype=float)
    for order, amplitude in enumerate(amplitudes, start=1):
        x += amplitude * np.cos(2 * np.pi * order * position)
    for frequency, amplitude, phase in constant:
        x += amplitude * np.cos(2 * np.pi * frequency * t + phase)

    fault_phase = fault_order * position
    crossing_count = np.floor(fault_phase).astype(np.int64)
    impacts = np.zeros(t.shape, dtype=float)
    impact_indices = np.flatnonzero(np.diff(crossing_count, prepend=0) > 0)
    impacts[impact_indices] = fault_amplitude

    kernel_t = np.arange(max(2, int(np.ceil(8 * resonance_decay * fs)))) / fs
    kernel = np.exp(-kernel_t / resonance_decay) * np.sin(
        2 * np.pi * resonance_frequency * kernel_t
    )
    peak = np.max(np.abs(kernel))
    if peak:
        kernel = kernel / peak
    x += lfilter(kernel, [1.0], impacts)
    if noise_std:
        x += np.random.default_rng(seed).normal(0.0, noise_std, size=len(t))
    return t, speed, position, x
