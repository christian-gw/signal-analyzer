"""Public package surface for the DSP scratchpad."""

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

__version__ = "0.1.0"
