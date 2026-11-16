import doctest

import pytest

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


@pytest.mark.parametrize(
    "module",
    [
        fft_tools,
        demodulation,
        indicators,
        sampling,
        synthetic_signals,
        resampling,
        measurement,
        analysis,
        tabular,
        file_utils,
    ],
    ids=lambda module: module.__name__.rsplit(".", 1)[-1],
)
def test_module_doctests(module):
    failures, _ = doctest.testmod(module, raise_on_error=False)
    assert failures == 0
