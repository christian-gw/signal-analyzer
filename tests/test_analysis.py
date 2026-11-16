import numpy as np
import pytest

from signal_analyzer.analysis import Analysis
from signal_analyzer.measurement import AnalysisResult, Measurement, Signal
from signal_analyzer.synthetic_signals import simulate_bearing_fault


def _tone_measurement():
    fs = 1024.0
    t = np.arange(2048) / fs
    x = 2.0 * np.sin(2 * np.pi * 64 * t)
    measurement = Measurement()
    measurement.add_signal(
        "x", Signal(x, sample_rate=fs, value_unit="g", coordinate_unit="s")
    )
    return measurement


def test_analysis_amplitude_psd_and_indicator_dispatch():
    measurement = _tone_measurement()
    analysis = Analysis(measurement)

    amplitude = analysis.amplitude_spectrum("x", "x_amplitude", window=None)
    psd = analysis.psd("x", "x_psd", window=None)
    amplitude_stats = analysis.amplitude_band_statistics(
        "x_amplitude", lower=60, upper=68
    )
    psd_stats = analysis.psd_band_statistics("x_psd", lower=60, upper=68)
    signal_stats = analysis.signal_statistics("x", lower=0.0, upper=1.0)

    assert amplitude.kind == "amplitude_spectrum"
    assert psd.kind == "psd"
    assert amplitude.metadata["normalization"] == "coherent_gain"
    assert amplitude.metadata["aggregation"] == "none"
    assert psd.metadata["normalization"] == "density"
    assert psd.metadata["aggregation"] == "none"
    assert "spectrum_kind" not in amplitude.metadata
    assert "domain" not in amplitude.metadata
    assert "source_coordinate_unit" not in amplitude.metadata
    assert "spectrum_kind" not in psd.metadata
    assert "domain" not in psd.metadata
    assert "source_coordinate_unit" not in psd.metadata
    assert isinstance(amplitude_stats, AnalysisResult)
    assert amplitude_stats.values["peak_coordinate"] == pytest.approx(64.0)
    assert amplitude_stats.values["peak_amplitude"] == pytest.approx(2.0)
    assert amplitude_stats.units["peak_amplitude"] == "g"
    assert amplitude_stats.units["peak_coordinate"] == "Hz"
    assert psd_stats.values["band_power"] == pytest.approx(2.0)
    assert psd_stats.values["band_rms"] == pytest.approx(np.sqrt(2.0))
    assert psd_stats.units["band_power"] == "(g)^2"
    assert psd_stats.units["band_rms"] == "g"
    assert signal_stats.values["rms"] == pytest.approx(np.sqrt(2.0))
    assert signal_stats.units["rms"] == "g"
    assert signal_stats.units["crest_factor"] == "1"

    with pytest.raises(ValueError, match="kind='psd'"):
        analysis.psd_band_statistics("x_amplitude")
    with pytest.raises(ValueError, match="kind='signal'"):
        analysis.downsample("x_psd", "invalid", down=2)


def test_analysis_downsample_adds_signal_and_provenance():
    measurement = _tone_measurement()
    analysis = Analysis(measurement)

    result = analysis.downsample("x", "x_down", down=4)

    assert result.sample_rate == 256.0
    assert len(result.values) == 512
    assert measurement.provenance[-1].operation == "resample_uniform"


def test_analysis_statistics_can_store_named_results_with_provenance():
    measurement = _tone_measurement()
    analysis = Analysis(measurement)

    result = analysis.signal_statistics(
        "x", lower=0.25, upper=1.25, output_name="x_statistics"
    )

    assert measurement.result("x_statistics") is result
    assert measurement.provenance[-1].operation == "signal_statistics"
    assert measurement.provenance[-1].inputs == ("x",)
    assert measurement.provenance[-1].parameters == {
        "lower": 0.25,
        "upper": 1.25,
    }


def test_analysis_statistics_keep_unknown_units_explicit():
    measurement = Measurement()
    measurement.add_signal("x", Signal(np.arange(8.0), sample_rate=2.0))

    result = Analysis(measurement).signal_statistics("x")

    assert result.units["rms"] is None
    assert result.units["mean_square"] is None
    assert result.units["energy"] is None
    assert result.units["crest_factor"] == "1"


def test_analysis_requires_synchronized_paired_signals():
    measurement = _tone_measurement()
    measurement.add_signal(
        "speed", Signal(np.ones(2048), sample_rate=1024.0, start=0.1)
    )

    with pytest.raises(ValueError, match="synchronized"):
        Analysis(measurement).spatial_resample("x", "speed", "spatial")


def test_analysis_requires_matching_paired_coordinate_units():
    measurement = _tone_measurement()
    measurement.add_signal(
        "speed",
        Signal(
            np.ones(2048),
            sample_rate=1024.0,
            value_unit="rev/s",
            coordinate_unit="sample",
        ),
    )

    with pytest.raises(ValueError, match="matching coordinate units"):
        Analysis(measurement).spatial_resample("x", "speed", "spatial")


def test_analysis_align_pair_crops_and_interpolates_explicitly():
    measurement = Measurement()
    reference_coordinate = np.arange(0.0, 5.0, 0.5)
    source_coordinate = np.arange(1.0, 4.1, 1.0)
    measurement.add_signal(
        "vibration",
        Signal(
            reference_coordinate**2,
            coordinate=reference_coordinate,
            value_unit="g",
            coordinate_unit="s",
        ),
    )
    measurement.add_signal(
        "speed",
        Signal(
            10 + source_coordinate,
            coordinate=source_coordinate,
            value_unit="rev/s",
            coordinate_unit="s",
        ),
    )

    vibration, speed = Analysis(measurement).align_pair(
        "vibration",
        "speed",
        "vibration_aligned",
        "speed_aligned",
    )

    np.testing.assert_array_equal(vibration.coordinates, np.arange(1.0, 4.1, 0.5))
    np.testing.assert_allclose(speed.values, 10 + speed.coordinates)
    np.testing.assert_array_equal(vibration.coordinates, speed.coordinates)
    assert measurement.provenance[-2].operation == "crop_to_coordinate_overlap"
    assert measurement.provenance[-1].operation == "align_to_reference"


def test_analysis_records_spectral_normalization_and_aggregation():
    measurement = _tone_measurement()
    analysis = Analysis(measurement)

    amplitude = analysis.amplitude_spectrum(
        "x", "x_amplitude_average", block_size=512
    )
    psd = analysis.psd("x", "x_psd_average", block_size=512)

    assert amplitude.metadata["normalization"] == "coherent_gain"
    assert amplitude.metadata["aggregation"] == "mean_magnitude"
    assert psd.metadata["normalization"] == "density"
    assert psd.metadata["aggregation"] == "mean_power_density"


def test_analysis_spatial_resample_recovers_known_order():
    fs = 1024.0
    t = np.arange(2048) / fs
    speed_values = np.full(t.shape, 8.0)
    x = np.sin(2 * np.pi * 3 * 8 * t)
    measurement = Measurement()
    measurement.add_signal(
        "x", Signal(x, sample_rate=fs, value_unit="g", coordinate_unit="s")
    )
    measurement.add_signal(
        "speed",
        Signal(speed_values, sample_rate=fs, value_unit="rev/s", coordinate_unit="s"),
    )
    analysis = Analysis(measurement, resolution=64)

    analysis.spatial_resample(
        "x", "speed", "x_spatial", position_unit="rev"
    )
    spectrum = analysis.amplitude_spectrum(
        "x_spatial", "order_spectrum", block_size=512, window="hann"
    )
    peak = spectrum.coordinates[np.argmax(spectrum.values[1:]) + 1]

    assert peak == pytest.approx(3.0, abs=64 / 512)
    assert spectrum.coordinate_unit == "cycles/rev"


def test_envelope_order_spectrum_happy_path_recovers_fault_order():
    fs = 4096.0
    fault_order = 4.8
    _, speed, _, x = simulate_bearing_fault(
        fs=fs,
        duration=5.0,
        speed_min=8.0,
        speed_max=16.0,
        shaft_order_amplitudes=(0.2, 0.1, 0.05, 0.03, 0.02),
        constant_components=(),
        fault_order=fault_order,
        fault_amplitude=1.0,
        resonance_frequency=1000.0,
        resonance_decay=0.003,
        noise_std=0.01,
        seed=5,
    )
    measurement = Measurement()
    measurement.add_signal(
        "vibration",
        Signal(x, sample_rate=fs, value_unit="g", coordinate_unit="s"),
    )
    measurement.add_signal(
        "speed",
        Signal(speed, sample_rate=fs, value_unit="rev/s", coordinate_unit="s"),
    )
    analysis = Analysis(measurement, resolution=64, oversample=4)

    spectrum = analysis.envelope_order_spectrum(
        "vibration",
        "speed",
        "bearing",
        band=(800.0, 1200.0),
        lowpass=200.0,
        block_size=512,
        trim_samples=int(0.5 * fs),
    )
    search = (spectrum.coordinates >= 3) & (spectrum.coordinates <= 7)
    measured_order = spectrum.coordinates[search][
        np.argmax(spectrum.values[search])
    ]

    assert measured_order == pytest.approx(fault_order, abs=64 / 512)
    assert spectrum.kind == "amplitude_spectrum"
    assert "bearing_envelope" in measurement.signals
    assert "bearing_spatial" in measurement.signals
    assert measurement.provenance[-1].operation == "averaged_fft"
