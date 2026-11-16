# `signal_analyzer` — short guide

This short guide is suitable for pasting into a coding chat when generating
scripts with this package. It describes the public API and the distinctions
that are easy to get wrong.

## Purpose and imports

Small NumPy/SciPy toolkit for vibration, temporal-frequency, order, spatial,
and envelope analysis. Install with `python -m pip install .`.

```python
from signal_analyzer import (
    Analysis, AnalysisResult, ChannelSpec, Measurement, Signal,
    measurement_from_dataframe,
)
```

Numerical functions can also be imported directly from
`signal_analyzer.fft_tools`, `.demodulation`, `.indicators`, `.sampling`, and
`.resampling`. Inputs and outputs are normally 1-D NumPy arrays.

## Data model

```python
m = Measurement(metadata={"source": "optional caller metadata"})
m.add_signal("x", Signal(
    values, sample_rate=fs, start=0.0,
    value_unit="g", coordinate_unit="s", kind="signal",
))
a = Analysis(m, resolution=64, oversample=4)
```

`Signal` requires exactly one grid description:

- uniform: `sample_rate` plus optional `start`;
- explicit: `coordinate=array` (finite and strictly increasing).

Use `None` for an unknown unit and `"1"` for dimensionless. Units are labels,
not converted or inferred. Access `signal.values`, `signal.coordinates`,
`signal.value_unit`, `signal.coordinate_unit`, and `signal.kind`.

`Measurement` holds named `signals`, scalar `results`, `metadata`, and
`provenance`. Use `m.signal(name)` and `m.result(name)`. Signal and result names
share one namespace. Source/file/job identity belongs in caller metadata, not in
a required measurement ID.

Analysis operations store derived signals. Indicator methods return an
`AnalysisResult`; pass `output_name=` only when it should also be stored with
provenance. Read it with `result.values["rms"]` and
`result.units["rms"]`.

## DataFrames

The package accepts a pandas DataFrame but does no file I/O:

```python
import pandas as pd
from signal_analyzer import ChannelSpec, measurement_from_dataframe

df = pd.read_parquet(path)                 # loading stays in caller code
m = measurement_from_dataframe(
    df,
    coordinate_column="time",              # OR use_index=True OR sample_rate=fs
    coordinate_unit="s",
    channels={
        "vibration": ChannelSpec("accel", value_unit="g"),
        "speed": ChannelSpec("rpm", value_unit="rpm"),
    },
)
```

Choose exactly one coordinate source. If `channels=None`, numeric
non-coordinate columns are selected automatically; an explicit mapping is
safer. The adapter is for one shared grid. Construct `Measurement` directly for
independent channel grids. Iterate over multiple files in caller code.

## Measurement-aware operations

```python
a.demodulate("x", "envelope", band=(800, 1200), lowpass=200,
             detector="hilbert", filter_order=4)
a.downsample("x", "x_low_rate", down=4, up=1)
a.align_pair("reference", "other", "reference_overlap", "other_aligned")
a.spatial_resample("x", "speed", "x_spatial",
                   resolution=64, oversample=4, position_unit="rev")
a.amplitude_spectrum("x", "x_amplitude", block_size=1024,
                     overlap=0.5, window="hann")
a.psd("x", "x_psd", block_size=1024, overlap=0.5,
      window="hann", detrend=False)

r1 = a.signal_statistics("x", lower=0, upper=2)
r2 = a.amplitude_band_statistics("x_amplitude", lower=10, upper=20)
r3 = a.psd_band_statistics("x_psd", lower=10, upper=20)
```

`align_pair` crops both signals to their coordinate overlap and linearly
interpolates the second onto the reference grid. It neither extrapolates nor
anti-aliases. Paired spatial operations otherwise require synchronized grids,
rates, and coordinate units.

The convenience pipeline
`a.envelope_order_spectrum(vibration_name, speed_name, output_prefix, band,
lowpass, block_size, trim_samples=..., ...)` performs demodulation, edge
trimming, speed-based spatial resampling, and an averaged amplitude spectrum.

## Direct numerical API

```python
from signal_analyzer.demodulation import amplitude_demodulate
from signal_analyzer.fft_tools import (
    fft_spectrum, averaged_fft, psd_spectrum, averaged_psd, power_spectrum,
)
from signal_analyzer.indicators import (
    signal_statistics, amplitude_band_statistics, psd_band_statistics,
)
from signal_analyzer.sampling import align_to_reference, resample_uniform
from signal_analyzer.resampling import speed_resample_aa
```

- `amplitude_demodulate(x, fs, band, lowpass, detector="hilbert")`
- `fft_spectrum(x, fs=..., window="hann") -> coordinate, complex_coefficients`
- `averaged_fft(x, block_size, fs=..., overlap=0.5) -> coordinate, amplitude`
- `psd_spectrum(...)` / `averaged_psd(...) -> frequency, PSD density`
- `power_spectrum(...) -> frequency, power per bin`
- `resample_uniform(x, fs, up=1, down=...) -> values, output_fs`
- `speed_resample_aa(x, speed, fs, resolution=64, oversample=4) -> position, values`

Standalone indicator functions return plain dictionaries; the caller owns unit
handling. Synthetic helpers live in `signal_analyzer.synthetic_signals`:
`synthesize_tone_mix` and `simulate_bearing_fault`.

## Critical semantics

- `averaged_fft` averages magnitudes. It is an amplitude spectrum, not power or
  PSD. Never derive band power or band RMS from it.
- PSD is power density; integrating/summing `PSD * bin_width` estimates mean
  square. `power_spectrum` is power per bin.
- Indicator ranges are half-open: `lower <= coordinate < upper`.
- Demodulate before spatial resampling when the carrier band is specified in
  Hz. Trim forward/backward-filter edge transients explicitly.
- Hilbert-envelope output keeps the input unit; square-law output has a squared
  unit.
- Speed must be finite and strictly positive. Stationary or reversing sections
  must be segmented or rejected, never silently clamped.
- Spatial resampling combines a fixed max-speed temporal filter, interpolation,
  and polyphase decimation. It is a partial anti-alias safeguard, not a rigorous
  guarantee. `resolution` is samples per integrated speed unit. Nyquist is
  `resolution / 2`; use roughly `resolution / 2.56` as the practical working
  boundary. Choose `oversample` with the maximum/minimum speed ratio in mind.
- Persistence, plotting, GUI logic, file readers, and multi-file orchestration
  stay outside this numerical/in-memory core.

When generating code, use only these public APIs, preserve explicit units and
grid semantics, and do not reinterpret amplitude spectra as power spectra.
