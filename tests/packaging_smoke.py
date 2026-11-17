"""Smoke-test an installed distribution from outside the source tree."""

import numpy as np

from signal_analyzer import Analysis, Measurement, Signal
from signal_analyzer import (
    analysis,
    demodulation,
    fft_tools,
    file_utils,
    indicators,
    measurement,
    resampling,
    sampling,
    synthetic_signals,
    tabular,
)


def main():
    public_modules = (
        analysis,
        demodulation,
        fft_tools,
        file_utils,
        indicators,
        measurement,
        resampling,
        sampling,
        synthetic_signals,
        tabular,
    )
    assert all(module.__package__ == "signal_analyzer" for module in public_modules)

    measurement_object = Measurement()
    measurement_object.add_signal(
        "x",
        Signal(
            np.arange(8, dtype=float),
            sample_rate=8.0,
            value_unit="g",
            coordinate_unit="s",
        ),
    )
    result = Analysis(measurement_object).signal_statistics("x")
    assert result.values["count"] == 8
    assert result.units["rms"] == "g"


if __name__ == "__main__":
    main()
