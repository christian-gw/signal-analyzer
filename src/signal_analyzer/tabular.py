"""Pandas DataFrame adapter for the in-memory measurement model."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .measurement import Measurement, Signal


@dataclass(frozen=True)
class ChannelSpec:
    """Describe one output signal selected from a wide DataFrame column."""

    column: str
    value_unit: str | None = None
    kind: str = "signal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("column must be a non-empty string")
        if self.value_unit is not None and (
            not isinstance(self.value_unit, str) or not self.value_unit.strip()
        ):
            raise ValueError("value_unit must be None or a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        try:
            metadata = dict(self.metadata)
        except (TypeError, ValueError):
            raise ValueError("metadata must be a mapping") from None
        object.__setattr__(self, "metadata", metadata)


def _numeric_values(values, value_name):
    if pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"{value_name} must contain numeric values")
    try:
        numeric = pd.to_numeric(values, errors="raise")
        result = np.asarray(numeric, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{value_name} must contain numeric values") from None
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{value_name} must contain at least one value")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{value_name} must contain only finite values")
    return result


def _channel_specs(frame, coordinate_column, channels):
    if channels is None:
        selected = {}
        for column in frame.columns:
            if column == coordinate_column:
                continue
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series.dtype) and not (
                pd.api.types.is_bool_dtype(series.dtype)
            ):
                selected[column] = ChannelSpec(column)
        if not selected:
            raise ValueError("DataFrame contains no numeric signal columns")
        return selected

    try:
        items = list(channels.items())
    except AttributeError:
        raise ValueError("channels must be a mapping or None") from None
    if not items:
        raise ValueError("channels must select at least one signal")

    selected = {}
    input_columns = set()
    for output_name, specification in items:
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("channel names must be non-empty strings")
        if isinstance(specification, str):
            specification = ChannelSpec(specification)
        if not isinstance(specification, ChannelSpec):
            raise ValueError("channel specifications must be strings or ChannelSpec")
        if specification.column == coordinate_column:
            raise ValueError("the coordinate column cannot also be a signal channel")
        if specification.column in input_columns:
            raise ValueError("each DataFrame signal column may be selected only once")
        input_columns.add(specification.column)
        selected[output_name] = specification
    return selected


def measurement_from_dataframe(
    frame,
    *,
    coordinate_column=None,
    use_index=False,
    sample_rate=None,
    start=0.0,
    channels=None,
    coordinate_unit=None,
    metadata=None,
):
    """Convert one numeric wide DataFrame into one shared-grid Measurement.

    Exactly one coordinate source is required: ``coordinate_column``,
    ``use_index=True``, or ``sample_rate``. ``channels`` maps output signal
    names to DataFrame column names or :class:`ChannelSpec` objects. When it is
    ``None``, every numeric non-coordinate column is selected; an explicit
    mapping is safer when a table contains numeric identifiers or metadata.

    Units are never inferred. Omitted value or coordinate units remain
    unknown. This adapter performs no file I/O and does not cover independent
    channel grids; callers can construct a Measurement directly for those
    cases.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("DataFrame column names must be unique")
    if not isinstance(use_index, (bool, np.bool_)):
        raise ValueError("use_index must be boolean")

    has_coordinate = coordinate_column is not None
    has_rate = sample_rate is not None
    if int(has_coordinate) + int(bool(use_index)) + int(has_rate) != 1:
        raise ValueError(
            "provide exactly one of coordinate_column, use_index=True, "
            "or sample_rate"
        )
    if has_coordinate and (
        not isinstance(coordinate_column, str) or not coordinate_column
    ):
        raise ValueError("coordinate_column must be a non-empty string")
    if (has_coordinate or use_index) and start != 0.0:
        raise ValueError("start is only valid with sample_rate")

    coordinate = None
    if has_coordinate:
        if coordinate_column not in frame.columns:
            raise ValueError(f"unknown DataFrame column: {coordinate_column}")
        coordinate = _numeric_values(
            frame[coordinate_column], "coordinate column"
        )
    elif use_index:
        coordinate = _numeric_values(frame.index.to_numpy(), "DataFrame index")
    if coordinate is not None and coordinate.size > 1:
        if np.any(np.diff(coordinate) <= 0):
            raise ValueError("coordinate must be strictly increasing")

    selected = _channel_specs(frame, coordinate_column, channels)
    measurement = Measurement(metadata={} if metadata is None else metadata)
    for output_name, specification in selected.items():
        if specification.column not in frame.columns:
            raise ValueError(f"unknown DataFrame column: {specification.column}")
        values = _numeric_values(
            frame[specification.column], f"channel {output_name!r}"
        )
        signal_kwargs = {
            "value_unit": specification.value_unit,
            "coordinate_unit": coordinate_unit,
            "kind": specification.kind,
            "metadata": specification.metadata,
        }
        if coordinate is not None:
            signal_kwargs["coordinate"] = coordinate
        else:
            signal_kwargs.update(sample_rate=sample_rate, start=start)
        measurement.add_signal(
            output_name,
            Signal(values, **signal_kwargs),
        )
    return measurement
