"""Public package surface for the DSP scratchpad."""

from importlib.metadata import version as _distribution_version

from .analysis import Analysis
from .measurement import AnalysisResult, Measurement, OperationRecord, Signal
from .tabular import ChannelSpec, measurement_from_dataframe

__all__ = [
    "Analysis",
    "AnalysisResult",
    "ChannelSpec",
    "Measurement",
    "OperationRecord",
    "Signal",
    "measurement_from_dataframe",
]

__version__ = _distribution_version("signal-analyzer")
