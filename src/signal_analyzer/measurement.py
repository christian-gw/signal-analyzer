"""Minimal in-memory signals, measurements, and provenance records."""

from numbers import Real
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ._validation import as_finite_1d_array, require_positive_finite_scalar


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        finite = bool(np.isfinite(value))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite scalar")


def _optional_unit(value, name):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be None or a non-empty string")
    return value


@dataclass(frozen=True)
class Signal:
    """A finite 1D signal on either a uniform or explicit coordinate grid.

    Exactly one of ``sample_rate`` or ``coordinate`` must be supplied. Uniform
    signals use ``start + arange(N) / sample_rate``; explicit coordinates may
    be nonuniform but must be finite and strictly increasing. Input arrays are
    copied and the stored arrays are read-only.
    """

    values: np.ndarray
    sample_rate: float | None = None
    coordinate: np.ndarray | None = None
    start: float = 0.0
    value_unit: str | None = None
    coordinate_unit: str | None = None
    kind: str = "signal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        values = as_finite_1d_array(self.values, "values").copy()
        if (self.sample_rate is None) == (self.coordinate is None):
            raise ValueError("provide exactly one of sample_rate or coordinate")
        _finite_scalar(self.start, "start")
        value_unit = _optional_unit(self.value_unit, "value_unit")
        coordinate_unit = _optional_unit(
            self.coordinate_unit, "coordinate_unit"
        )
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")

        coordinate = None
        sample_rate = self.sample_rate
        if sample_rate is not None:
            require_positive_finite_scalar(sample_rate, "sample_rate")
            sample_rate = float(sample_rate)
        else:
            coordinate = as_finite_1d_array(
                self.coordinate, "coordinate", min_size=values.size, real=True
            ).copy()
            if coordinate.size != values.size:
                raise ValueError("coordinate and values must have equal length")
            if coordinate.size > 1 and np.any(np.diff(coordinate) <= 0):
                raise ValueError("coordinate must be strictly increasing")

        values.setflags(write=False)
        if coordinate is not None:
            coordinate.setflags(write=False)

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "value_unit", value_unit)
        object.__setattr__(self, "coordinate_unit", coordinate_unit)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def coordinates(self):
        """Return coordinate values without forcing unrelated signals to align."""
        if self.coordinate is not None:
            return self.coordinate.copy()
        return self.start + np.arange(len(self.values)) / self.sample_rate

    def inferred_sample_rate(self):
        """Return the uniform sample rate, or None for a nonuniform grid."""
        if self.sample_rate is not None:
            return self.sample_rate
        if len(self.coordinate) < 2:
            return None
        differences = np.diff(self.coordinate)
        spacing = differences[0]
        tolerance = 32 * np.finfo(float).eps * max(1.0, abs(spacing))
        if not np.allclose(differences, spacing, rtol=1e-10, atol=tolerance):
            return None
        return float(1 / spacing)


@dataclass(frozen=True)
class AnalysisResult:
    """Named scalar analysis values with explicit per-value units.

    ``None`` denotes an unknown unit and ``"1"`` denotes an explicitly
    dimensionless value. NaN values are accepted for meaningful undefined
    statistics, while infinite values are rejected.
    """

    kind: str
    values: Mapping[str, int | float]
    units: Mapping[str, str | None]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        try:
            values = dict(self.values)
            units = dict(self.units)
        except (TypeError, ValueError):
            raise ValueError("values and units must be mappings") from None
        if not values:
            raise ValueError("values must not be empty")
        if not all(isinstance(name, str) and name for name in values):
            raise ValueError("values keys must be non-empty strings")
        if units.keys() != values.keys():
            raise ValueError("units must contain exactly the values keys")

        checked_values = {}
        for name, value in values.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("values must contain real numeric scalars")
            if np.isinf(value):
                raise ValueError("values must not contain infinity")
            checked_values[name] = (
                int(value) if isinstance(value, (int, np.integer)) else float(value)
            )
        checked_units = {
            name: _optional_unit(unit, f"units[{name!r}]")
            for name, unit in units.items()
        }

        object.__setattr__(self, "values", checked_values)
        object.__setattr__(self, "units", checked_units)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class OperationRecord:
    """Serializable-style provenance for one derived artifact operation."""

    operation: str
    inputs: tuple[str, ...]
    output: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("output must be a non-empty string")
        inputs = tuple(self.inputs)
        if not all(isinstance(name, str) and name for name in inputs):
            raise ValueError("inputs must contain non-empty signal names")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass
class Measurement:
    """Named signals and scalar results with metadata and provenance."""

    signals: dict[str, Signal] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: list[OperationRecord] = field(default_factory=list)
    results: dict[str, AnalysisResult] = field(default_factory=dict)

    def __post_init__(self):
        original_signals = dict(self.signals)
        original_results = dict(self.results)
        self.signals = {}
        self.results = {}
        for name, signal in original_signals.items():
            self.add_signal(name, signal)
        for name, result in original_results.items():
            self.add_result(name, result)
        self.metadata = dict(self.metadata)
        self.provenance = list(self.provenance)

    def _name_exists(self, name):
        return name in self.signals or name in self.results

    def add_signal(self, name, signal):
        """Add an original signal; duplicate names are rejected."""
        if not isinstance(name, str) or not name:
            raise ValueError("signal name must be a non-empty string")
        if not isinstance(signal, Signal):
            raise ValueError("signal must be a Signal")
        if self._name_exists(name):
            raise ValueError(f"name already exists: {name}")
        self.signals[name] = signal
        return signal

    def add_result(self, name, result):
        """Add a scalar result; names are shared with signals."""
        if not isinstance(name, str) or not name:
            raise ValueError("result name must be a non-empty string")
        if not isinstance(result, AnalysisResult):
            raise ValueError("result must be an AnalysisResult")
        if self._name_exists(name):
            raise ValueError(f"name already exists: {name}")
        self.results[name] = result
        return result

    def add_derived_signal(
        self, name, signal, operation, inputs, parameters=None
    ):
        """Atomically add a derived signal and its operation record."""
        inputs = tuple(inputs)
        missing = [input_name for input_name in inputs if input_name not in self.signals]
        if missing:
            raise ValueError(f"unknown input signals: {missing}")
        record = OperationRecord(
            operation=operation,
            inputs=inputs,
            output=name,
            parameters={} if parameters is None else parameters,
        )
        self.add_signal(name, signal)
        self.provenance.append(record)
        return signal

    def add_derived_result(
        self, name, result, operation, inputs, parameters=None
    ):
        """Atomically add a derived scalar result and its operation record."""
        inputs = tuple(inputs)
        missing = [input_name for input_name in inputs if not self._name_exists(input_name)]
        if missing:
            raise ValueError(f"unknown inputs: {missing}")
        record = OperationRecord(
            operation=operation,
            inputs=inputs,
            output=name,
            parameters={} if parameters is None else parameters,
        )
        self.add_result(name, result)
        self.provenance.append(record)
        return result

    def signal(self, name):
        """Return a named signal with a clear error for missing names."""
        try:
            return self.signals[name]
        except KeyError:
            raise ValueError(f"unknown signal: {name}") from None

    def result(self, name):
        """Return a named scalar result with a clear error when missing."""
        try:
            return self.results[name]
        except KeyError:
            raise ValueError(f"unknown result: {name}") from None
