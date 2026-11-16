import numpy as np
import pytest

from signal_analyzer.measurement import (
    AnalysisResult,
    Measurement,
    OperationRecord,
    Signal,
)


def test_uniform_signal_generates_coordinates_and_rate():
    signal = Signal(
        np.arange(4), sample_rate=2.0, start=1.5,
        value_unit="m/s2", coordinate_unit="s",
    )

    np.testing.assert_allclose(signal.coordinates, [1.5, 2.0, 2.5, 3.0])
    assert signal.inferred_sample_rate() == 2.0


def test_explicit_uniform_and_nonuniform_coordinate_rate_inference():
    uniform = Signal(np.ones(3), coordinate=np.array([0.0, 0.25, 0.5]))
    nonuniform = Signal(np.ones(3), coordinate=np.array([0.0, 0.2, 0.5]))

    assert uniform.inferred_sample_rate() == 4.0
    assert nonuniform.inferred_sample_rate() is None


def test_signal_distinguishes_unknown_and_dimensionless_units():
    unknown = Signal(np.ones(3), sample_rate=1.0)
    dimensionless = Signal(
        np.ones(3), sample_rate=1.0, value_unit="1", coordinate_unit="s"
    )

    assert unknown.value_unit is None
    assert unknown.coordinate_unit is None
    assert dimensionless.value_unit == "1"

    for invalid in ["", "   ", 1]:
        with pytest.raises(ValueError, match="value_unit"):
            Signal(np.ones(3), sample_rate=1.0, value_unit=invalid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"sample_rate": 10.0, "coordinate": np.arange(3)},
        {"coordinate": np.array([0.0, 0.0, 1.0])},
        {"coordinate": np.arange(2)},
        {"sample_rate": 0.0},
        {"sample_rate": 10.0, "start": np.nan},
    ],
)
def test_signal_rejects_invalid_coordinate_contract(kwargs):
    with pytest.raises(ValueError):
        Signal(np.ones(3), **kwargs)


def test_measurement_preserves_independent_time_bases():
    measurement = Measurement()
    fast = Signal(np.ones(10), sample_rate=100.0, coordinate_unit="s")
    slow = Signal(np.ones(4), sample_rate=10.0, start=2.0, coordinate_unit="s")

    measurement.add_signal("fast", fast)
    measurement.add_signal("slow", slow)

    assert measurement.signal("fast").sample_rate == 100.0
    assert measurement.signal("slow").start == 2.0


def test_measurement_rejects_duplicates_and_unknown_names():
    measurement = Measurement()
    measurement.add_signal("x", Signal(np.ones(4), sample_rate=1.0))

    with pytest.raises(ValueError, match="already exists"):
        measurement.add_signal("x", Signal(np.ones(4), sample_rate=1.0))
    with pytest.raises(ValueError, match="unknown signal"):
        measurement.signal("missing")


def test_derived_signal_records_plain_provenance():
    measurement = Measurement(metadata={"case": "synthetic"})
    measurement.add_signal("x", Signal(np.ones(4), sample_rate=2.0))
    derived = Signal(np.arange(2), sample_rate=1.0)

    measurement.add_derived_signal(
        "x_down", derived, "resample_uniform", ("x",), {"down": 2}
    )

    assert measurement.signal("x_down") is derived
    assert measurement.provenance == [
        OperationRecord(
            "resample_uniform", ("x",), "x_down", {"down": 2}
        )
    ]


def test_derived_signal_requires_known_inputs():
    measurement = Measurement()

    with pytest.raises(ValueError, match="unknown input"):
        measurement.add_derived_signal(
            "y", Signal(np.ones(2), sample_rate=1.0), "copy", ("x",)
        )


def test_analysis_result_validates_and_copies_values_units_and_metadata():
    values = {"count": np.int64(4), "rms": np.float64(2.5), "skewness": np.nan}
    units = {"count": "1", "rms": None, "skewness": "1"}
    metadata = {"selection": "overall"}

    result = AnalysisResult("signal_statistics", values, units, metadata)
    values["rms"] = 99
    units["rms"] = "g"
    metadata["selection"] = "changed"

    assert result.values == {"count": 4, "rms": 2.5, "skewness": np.nan}
    assert isinstance(result.values["count"], int)
    assert result.units["rms"] is None
    assert result.metadata == {"selection": "overall"}


@pytest.mark.parametrize(
    "values, units, message",
    [
        ({}, {}, "must not be empty"),
        ({"x": True}, {"x": "1"}, "real numeric"),
        ({"x": 1 + 2j}, {"x": "1"}, "real numeric"),
        ({"x": np.inf}, {"x": "1"}, "infinity"),
        ({"x": 1.0}, {}, "exactly"),
        ({"x": 1.0}, {"x": ""}, "units"),
    ],
)
def test_analysis_result_rejects_invalid_contract(values, units, message):
    with pytest.raises(ValueError, match=message):
        AnalysisResult("statistics", values, units)


def test_measurement_stores_results_with_shared_names_and_lookup():
    measurement = Measurement()
    signal = Signal(np.ones(4), sample_rate=1.0)
    result = AnalysisResult("statistics", {"rms": 1.0}, {"rms": None})

    measurement.add_signal("x", signal)
    measurement.add_result("x_stats", result)

    assert measurement.result("x_stats") is result
    with pytest.raises(ValueError, match="already exists"):
        measurement.add_result("x", result)
    with pytest.raises(ValueError, match="already exists"):
        measurement.add_signal(
            "x_stats", Signal(np.ones(2), sample_rate=1.0)
        )
    with pytest.raises(ValueError, match="unknown result"):
        measurement.result("missing")


def test_derived_result_records_plain_provenance():
    measurement = Measurement()
    measurement.add_signal("x", Signal(np.ones(4), sample_rate=2.0))
    result = AnalysisResult(
        "signal_statistics", {"rms": 1.0}, {"rms": None}
    )

    measurement.add_derived_result(
        "x_stats",
        result,
        "signal_statistics",
        ("x",),
        {"lower": 0.0, "upper": 1.0},
    )

    assert measurement.result("x_stats") is result
    assert measurement.provenance[-1] == OperationRecord(
        "signal_statistics",
        ("x",),
        "x_stats",
        {"lower": 0.0, "upper": 1.0},
    )

    with pytest.raises(ValueError, match="unknown inputs"):
        measurement.add_derived_result(
            "missing_stats", result, "statistics", ("missing",)
        )


def test_measurement_instances_do_not_share_mutable_state():
    first = Measurement(metadata={"source": "first"})
    second = Measurement(metadata={"source": "second"})

    first.add_signal("x", Signal(np.ones(2), sample_rate=1.0))
    first.add_result(
        "stats", AnalysisResult("statistics", {"rms": 1.0}, {"rms": None})
    )

    assert set(first.signals) == {"x"}
    assert set(first.results) == {"stats"}
    assert second.signals == {}
    assert second.results == {}
    assert second.provenance == []
    assert second.metadata == {"source": "second"}
