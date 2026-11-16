import numpy as np
import pandas as pd
import pytest

from signal_analyzer.measurement import Measurement
from signal_analyzer.tabular import ChannelSpec, measurement_from_dataframe


def test_dataframe_with_explicit_coordinate_and_specs():
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.25],
            "acceleration": [1.0, 2.0, 3.0],
            "speed": [10.0, 11.0, 12.0],
            "label": ["a", "b", "c"],
        }
    )
    measurement = measurement_from_dataframe(
        frame,
        coordinate_column="time",
        coordinate_unit="s",
        channels={
            "vibration": ChannelSpec(
                "acceleration", value_unit="g", metadata={"axis": "radial"}
            ),
            "speed": ChannelSpec("speed", value_unit="rev/s"),
        },
        metadata={"case": "synthetic"},
    )

    assert isinstance(measurement, Measurement)
    assert measurement.metadata == {"case": "synthetic"}
    np.testing.assert_allclose(
        measurement.signal("vibration").coordinates, [0.0, 0.1, 0.25]
    )
    assert measurement.signal("vibration").value_unit == "g"
    assert measurement.signal("vibration").coordinate_unit == "s"
    assert measurement.signal("vibration").metadata == {"axis": "radial"}
    assert measurement.signal("speed").value_unit == "rev/s"


def test_dataframe_with_uniform_rate_and_unknown_units():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    measurement = measurement_from_dataframe(
        frame,
        sample_rate=20.0,
        start=1.5,
        channels={"x": "x"},
    )

    signal = measurement.signal("x")
    np.testing.assert_allclose(signal.coordinates, [1.5, 1.55, 1.6])
    assert signal.value_unit is None
    assert signal.coordinate_unit is None


def test_dataframe_index_can_supply_coordinate():
    frame = pd.DataFrame(
        {"x": [3.0, 4.0, 5.0]},
        index=pd.Index([0.2, 0.4, 0.9], name="time"),
    )

    measurement = measurement_from_dataframe(
        frame, use_index=True, coordinate_unit="s"
    )

    np.testing.assert_allclose(
        measurement.signal("x").coordinates, [0.2, 0.4, 0.9]
    )


def test_dataframe_auto_selects_numeric_columns_without_mutating_input():
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.5],
            "x": [1.0, 2.0],
            "speed": [3, 4],
            "label": ["first", "second"],
        }
    )
    original = frame.copy(deep=True)

    measurement = measurement_from_dataframe(
        frame, coordinate_column="time"
    )

    assert set(measurement.signals) == {"x", "speed"}
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        (
            {"coordinate_column": "time", "sample_rate": 10.0},
            "exactly one",
        ),
        (
            {"coordinate_column": "time", "use_index": True},
            "exactly one",
        ),
        (
            {"coordinate_column": "time", "start": 1.0},
            "start is only valid",
        ),
        (
            {"use_index": True, "start": 1.0},
            "start is only valid",
        ),
        (
            {"coordinate_column": "time", "channels": {}},
            "at least one",
        ),
        (
            {
                "coordinate_column": "time",
                "channels": {"time_signal": "time"},
            },
            "coordinate column",
        ),
        (
            {
                "coordinate_column": "time",
                "channels": {"first": "x", "second": "x"},
            },
            "only once",
        ),
    ],
)
def test_dataframe_adapter_rejects_invalid_contract(kwargs, message):
    frame = pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0]})

    with pytest.raises(ValueError, match=message):
        measurement_from_dataframe(frame, **kwargs)


@pytest.mark.parametrize(
    "frame, kwargs, message",
    [
        (
            pd.DataFrame({"time": [0.0, 0.0], "x": [1.0, 2.0]}),
            {"coordinate_column": "time", "channels": {"x": "x"}},
            "strictly increasing",
        ),
        (
            pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, np.nan]}),
            {"coordinate_column": "time", "channels": {"x": "x"}},
            "finite",
        ),
        (
            pd.DataFrame({"time": [0.0, 1.0], "x": ["one", "two"]}),
            {"coordinate_column": "time", "channels": {"x": "x"}},
            "numeric",
        ),
        (
            pd.DataFrame({"time": [0.0, 1.0], "label": ["a", "b"]}),
            {"coordinate_column": "time"},
            "no numeric signal",
        ),
        (
            pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0]}),
            {"coordinate_column": "time", "channels": {"x": "missing"}},
            "unknown DataFrame column",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0]}, index=[1.0, 1.0]),
            {"use_index": True},
            "strictly increasing",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"]),
            {"use_index": True},
            "numeric",
        ),
    ],
)
def test_dataframe_adapter_rejects_invalid_data(frame, kwargs, message):
    with pytest.raises(ValueError, match=message):
        measurement_from_dataframe(frame, **kwargs)


def test_dataframe_adapter_rejects_non_dataframe_and_duplicate_columns():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        measurement_from_dataframe({"x": [1, 2]}, sample_rate=1.0)

    duplicate = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        measurement_from_dataframe(duplicate, sample_rate=1.0)

    with pytest.raises(ValueError, match="use_index"):
        measurement_from_dataframe(
            pd.DataFrame({"x": [1.0]}), use_index="yes"
        )


def test_channel_spec_validates_and_copies_metadata():
    metadata = {"axis": "x"}
    specification = ChannelSpec("value", metadata=metadata)
    metadata["axis"] = "changed"
    assert specification.metadata == {"axis": "x"}

    with pytest.raises(ValueError, match="column"):
        ChannelSpec("")
    with pytest.raises(ValueError, match="value_unit"):
        ChannelSpec("x", value_unit="")
    with pytest.raises(ValueError, match="value_unit"):
        ChannelSpec("x", value_unit="   ")
