"""Synthetic study of the spatial resampling anti-aliasing stages.

Run from the repository root with::

    python -m examples.spatial_antialiasing_study

The high-resolution reference is densely sampled in time, interpolated to a
16x spatial grid, then polyphase-decimated to the target resolution. It is a
useful numerical reference for these synthetic signals, not a general proof of
alias-free nonuniform resampling.
"""

import numpy as np
from scipy.signal import resample_poly

from signal_analyzer.resampling import speed_resample, speed_resample_aa


def _spatial_only_aa(x, speed, fs, resolution, oversample):
    """Reproduce the earlier spatial oversample/decimate path."""
    pos, y = speed_resample(
        x, speed, fs, resolution=resolution * oversample
    )
    y = resample_poly(y, up=1, down=oversample, padtype="line")
    return pos[::oversample], y


def _tone_amplitude(x, pos, spatial_order):
    basis = np.exp(-2j * np.pi * spatial_order * pos)
    return 2 * np.abs(np.dot(x, basis)) / len(x)


def _rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _trimmed_pair(actual, reference):
    n = min(len(actual), len(reference))
    edge = max(8, int(0.05 * n))
    if 2 * edge >= n:
        edge = 0
    stop = n - edge if edge else n
    return actual[edge:stop], reference[edge:stop]


def _alias_order(spatial_order, resolution):
    wrapped = (spatial_order + resolution / 2) % resolution - resolution / 2
    return abs(wrapped)


def constant_speed_study():
    """Compare target-band amplitude and aliases at constant speed."""
    fs = 4096.0
    resolution = 64
    speed_value = 8.0
    duration = 1.25
    t = np.arange(int(duration * fs) + 1) / fs
    speed = np.full(t.shape, speed_value)
    orders = (16.0, resolution / 2.56, 36.0, 48.0)

    print("Constant speed: 8 unit/s, target Nyquist: 32 cycles/unit")
    print("Temporal max-speed cutoff: 256 Hz")
    print("order  method              measured amplitude  RMS error vs ideal")
    for spatial_order in orders:
        x = np.sin(2 * np.pi * spatial_order * speed_value * t)
        methods = {
            "direct": speed_resample(
                x, speed, fs, resolution=resolution
            ),
            "spatial-only x4": _spatial_only_aa(
                x, speed, fs, resolution, oversample=4
            ),
            "max-speed + x4": speed_resample_aa(
                x, speed, fs, resolution=resolution, oversample=4
            ),
        }
        measured_order = _alias_order(spatial_order, resolution)
        for name, (pos, y) in methods.items():
            if spatial_order < resolution / 2:
                ideal = np.sin(2 * np.pi * spatial_order * pos)
            else:
                ideal = np.zeros_like(y)
            actual_trimmed, ideal_trimmed = _trimmed_pair(y, ideal)
            amplitude = _tone_amplitude(y, pos, measured_order)
            error = _rms(actual_trimmed - ideal_trimmed)
            print(
                f"{spatial_order:5.1f}  {name:18s}  "
                f"{amplitude:18.5f}  {error:18.5f}"
            )
    print()


def _variable_speed_signal(
    fs, temporal_frequency, speed_min=1.0, speed_max=10.0, duration=2.0
):
    t = np.arange(int(duration * fs) + 1) / fs
    speed_mean = (speed_min + speed_max) / 2
    speed_amplitude = (speed_max - speed_min) / 2
    speed = speed_mean + speed_amplitude * np.sin(2 * np.pi * t / duration)
    x = np.sin(2 * np.pi * temporal_frequency * t)
    return x, speed


def _variable_speed_reference(
    temporal_frequency, resolution, speed_min, speed_max
):
    fs_reference = 65536.0
    x, speed = _variable_speed_signal(
        fs_reference, temporal_frequency, speed_min, speed_max
    )
    pos_high, y_high = speed_resample(
        x, speed, fs_reference, resolution=resolution * 16
    )
    y = resample_poly(y_high, up=1, down=16, padtype="line")
    return pos_high[::16], y


def variable_speed_study(
    temporal_frequency, speed_min=1.0, speed_max=10.0,
    oversamples=(2, 4, 8),
):
    """Compare errors for a fixed temporal tone under varying speed."""
    fs = 4096.0
    resolution = 64
    x, speed = _variable_speed_signal(
        fs, temporal_frequency, speed_min, speed_max
    )
    _, reference = _variable_speed_reference(
        temporal_frequency, resolution, speed_min, speed_max
    )

    methods = {
        "direct": speed_resample(x, speed, fs, resolution=resolution)[1],
    }
    for oversample in oversamples:
        methods[f"spatial-only x{oversample}"] = _spatial_only_aa(
            x, speed, fs, resolution, oversample
        )[1]
        methods[f"max-speed + x{oversample}"] = speed_resample_aa(
            x, speed, fs, resolution=resolution, oversample=oversample
        )[1]

    print(
        f"Variable speed: {speed_min:g}..{speed_max:g} unit/s, temporal tone: "
        f"{temporal_frequency:.0f} Hz"
    )
    print("Target Nyquist: 32 cycles/unit; max-speed cutoff: 320 Hz")
    _, reference_trimmed = _trimmed_pair(reference, reference)
    reference_rms = _rms(reference_trimmed)
    print(f"High-resolution reference RMS (trimmed): {reference_rms:.5f}")
    print("method              output RMS  RMS error  relative error")
    for name, y in methods.items():
        actual_trimmed, reference_trimmed = _trimmed_pair(y, reference)
        output_rms = _rms(actual_trimmed)
        error = _rms(actual_trimmed - reference_trimmed)
        relative = error / reference_rms if reference_rms >= 1e-3 else None
        relative_text = f"{relative:14.5f}" if relative is not None else "           n/a"
        print(
            f"{name:18s}  {output_rms:10.5f}  {error:9.5f}  "
            f"{relative_text}"
        )
    print()


def main():
    constant_speed_study()
    print("Intended 4:1 envelope; practical band edge is 64 / 2.56 = 25")
    variable_speed_study(125.0, speed_min=2.5, speed_max=10.0,
                         oversamples=(4,))
    variable_speed_study(250.0, speed_min=2.5, speed_max=10.0,
                         oversamples=(4,))
    variable_speed_study(300.0, speed_min=2.5, speed_max=10.0,
                         oversamples=(4,))

    print("10:1 stress cases")
    variable_speed_study(240.0)
    variable_speed_study(480.0)


if __name__ == "__main__":
    main()
