# DSP scratchpad

[![Tests](https://github.com/christian-gw/signal-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/christian-gw/signal-analyzer/actions/workflows/tests.yml)

A small, script-oriented NumPy/SciPy toolkit for exploratory vibration, temporal-frequency, order, spatial, and envelope analysis. A pandas DataFrame adapter is included for simple shared-grid data; file-format loaders and application-specific readers remain outside the package.

## Project status and scope

This is an exploratory, pre-1.0 DSP toolkit rather than a production framework.
Its public numerical functions, optional in-memory measurement layer, tests, and
synthetic examples are usable from notebooks and scripts. Persistence, plotting,
GUI behavior, application-specific readers, and multi-file orchestration remain
outside the core. API changes before 1.0 will be tested and documented, but
should not be assumed to be backward compatible.

## Installation

Install the package from a local checkout with:

```text
python -m pip install .
```

For development, install it in editable mode with the test and example
dependencies:

```text
python -m pip install -e ".[test,examples]"
```

The package requires Python 3.10 or newer. See [PACKAGING.md](PACKAGING.md) for
the source layout, regular and editable installation, and wheel-validation
workflow.

## Documentation

For a paste-ready API summary intended for token-limited coding chats, see the
[short guide](SHORT_GUIDE.md). It complements rather than duplicates the
detailed spectral and validation notes.

- [SPECTRAL_ANALYSIS.md](SPECTRAL_ANALYSIS.md) explains amplitude, power, PSD,
  windows, Welch averaging, and normalization.
- [PACKAGING.md](PACKAGING.md) covers installation and local package builds.
- [NOTES.md](NOTES.md) records numerical and design conventions.
- The two [example notebooks](examples) demonstrate the standalone and
  measurement-aware workflows on deterministic synthetic signals.

## Public API

| Module | Function | Purpose |
|---|---|---|
| `signal_analyzer.fft_tools` | `fft_spectrum`, `averaged_fft` | Complex amplitude FFT and blockwise mean-magnitude spectrum |
| `signal_analyzer.fft_tools` | `psd_spectrum`, `averaged_psd`, `power_spectrum` | PSD density and power-per-bin spectra |
| `signal_analyzer.demodulation` | `amplitude_demodulate` | Temporal band-pass, Hilbert-envelope or square-law detection, then low-pass |
| `signal_analyzer.indicators` | Range and statistics helpers | Half-open coordinate selection plus signal, amplitude-band, and PSD-band indicators |
| `signal_analyzer.sampling` | `resample_uniform`, `align_to_reference` | Anti-aliased uniform resampling and explicit overlap-only interpolation |
| `signal_analyzer.synthetic_signals` | Synthetic generators | Configurable tone mixes and a qualitative variable-speed bearing-fault workflow signal |
| `signal_analyzer.resampling` | Spatial resampling helpers | Speed integration, max-speed filtering, spatial interpolation, and polyphase decimation |
| `signal_analyzer.measurement` | `Signal`, `AnalysisResult`, `Measurement`, `OperationRecord` | Nullable units, independent grids, named artifacts, and provenance |
| `signal_analyzer.analysis` | `Analysis` | Measurement-aware composition with structured scalar results |
| `signal_analyzer.tabular` | `measurement_from_dataframe`, `ChannelSpec` | File-agnostic pandas DataFrame conversion for one shared grid |
| `signal_analyzer.file_utils` | `find_files` | Recursive file discovery sorted case-insensitively by filename |

Runtime dependencies are `numpy`, `scipy`, and `pandas`. The walkthrough notebooks additionally need `matplotlib` and a Jupyter environment. `nbclient` is only needed for headless notebook validation.

## Quick start

Run this example from the repository root:

```python
from signal_analyzer.demodulation import amplitude_demodulate
from signal_analyzer.fft_tools import averaged_fft
from signal_analyzer.resampling import speed_resample_aa
from signal_analyzer.synthetic_signals import simulate_bearing_fault

fs = 4096
t, speed, _, x = simulate_bearing_fault(
    fs=fs,
    duration=6,
    speed_min=5,
    speed_max=20,
    fault_order=4.9,
    resonance_frequency=1000,
    seed=3,
)

# The carrier band is temporal, so demodulate before position resampling.
envelope = amplitude_demodulate(
    x, fs=fs, band=(800, 1200), lowpass=200, detector="hilbert"
)

# Trim forward-backward filter edges, keeping speed synchronized with x.
edge = int(0.5 * fs)
pos, envelope_spatial = speed_resample_aa(
    envelope[edge:-edge], speed[edge:-edge], fs=fs, resolution=64
)
order, amplitude = averaged_fft(
    envelope_spatial, block_size=512, fs=64, window="hann", overlap=0.5
)
```

Here `order` is cycles per revolution because speed was supplied in revolutions per second. More generally, spatial FFT output is cycles per integrated speed unit. The returned spectrum is a mean amplitude spectrum, not a PSD.

## Optional measurement and analysis layer

The numerical functions remain directly usable. For named signals, independent time bases, and provenance, the same synthetic arrays can be wrapped without changing the algorithms:

```python
from signal_analyzer import Analysis, Measurement, Signal

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
order_spectrum = analysis.envelope_order_spectrum(
    "vibration",
    "speed",
    output_prefix="bearing",
    band=(800, 1200),
    lowpass=200,
    block_size=512,
    trim_samples=edge,
)
```

`Analysis` stores derived signals, optionally named scalar results, and plain operation records in the `Measurement`. It does not implement persistence or plotting.

Measurement-aware indicator methods return an `AnalysisResult` with per-value units. Supplying `output_name` also retains the result and its provenance:

```python
statistics = analysis.amplitude_band_statistics(
    "bearing_order_spectrum",
    lower=4,
    upper=6,
    output_name="fault_order_statistics",
)
print(statistics.values["peak_coordinate"], statistics.units["peak_coordinate"])
```

`None` means that a unit was not supplied; `"1"` means explicitly dimensionless. Standalone numerical indicator functions continue to return plain dictionaries, allowing callers to manage units themselves.

## Tabular input and multiple sources

`measurement_from_dataframe` maps one in-memory pandas DataFrame to one unnamed `Measurement`. It performs no file I/O and requires an explicit coordinate column, the DataFrame index, or a sample rate. Units are supplied through `ChannelSpec` and are never guessed. All selected channels share the chosen grid; construct `Signal` objects directly for independent grids.

```python
import pandas as pd

from signal_analyzer import ChannelSpec, measurement_from_dataframe

frame = pd.read_parquet("measurement.parquet")  # Loader choice is external.
measurement = measurement_from_dataframe(
    frame,
    coordinate_column="time",
    coordinate_unit="s",
    channels={
        "vibration": ChannelSpec("acceleration", value_unit="g"),
    },
)
```

Multiple sources remain an ordinary caller loop: load a DataFrame, convert it, and analyze it independently. The source path, GUI document, or job record provides external identity; `Measurement` deliberately has no file or batch identifier.

## Numerical conventions and limitations

- `x` denotes a signal and `Xf` complex FFT coefficients. With coherent-gain correction, a one-sided FFT bin estimates the peak amplitude of a bin-centered real sinusoid; a two-sided spectrum splits it across positive and negative frequencies.
- `averaged_fft` averages magnitudes, not complex coefficients or power. Its integer hop is floored, so effective overlap can be slightly larger than requested.
- PSD functions use window-power normalization and preserve DC unless detrending is requested. `power_spectrum` is PSD multiplied by bin width, so its sum estimates mean square.
- Signal RMS and energy are calculated from signal samples. Band power and band RMS require PSD input; they are never inferred from an averaged amplitude spectrum.
- Coordinate bands and time zones use half-open ranges: `lower <= coordinate < upper`.
- Signal and result units are nullable strings. Unknown units propagate as `None`; dimensionless metrics use `"1"`. The core does not guess or convert units.
- `resample_uniform` handles ordinary uniform-grid rational resampling separately from speed-to-position resampling.
- `align_to_reference` and `Analysis.align_pair` use linear interpolation on the actual coordinate overlap and never extrapolate. They are not substitutes for anti-aliased downsampling.
- `speed_resample` requires synchronous, strictly positive speed and returns a deterministic half-open position grid. The speed denominator must match the sampling-rate time unit—for example, samples/s with rev/s, not rpm—and the package does not parse or convert unit labels.
- `speed_resample_aa` provides partial anti-aliasing protection. Its fixed max-speed temporal filter and oversample/polyphase stage do not guarantee alias-free nonuniform reconstruction.
- The demonstrated working envelope uses a 4:1 speed ratio, default 4× spatial oversampling, and a practical analysis band ending at `resolution / 2.56`.
- Paired `Analysis` operations require equal coordinate grids, sample rates, and coordinate-unit labels. They do not align or convert channels implicitly.
- `Signal` copies its input arrays and stores them read-only. Derived signals copy source metadata; operation-specific metadata such as spectral normalization takes precedence.
- Hilbert demodulation retains amplitude units; square-law demodulation returns squared units. Both use zero-phase filters with endpoint transients.
- `simulate_bearing_fault` is a qualitative workflow signal, not a calibrated mechanical model.

See [SPECTRAL_ANALYSIS.md](SPECTRAL_ANALYSIS.md) for a concise explanation of spectrum types, windows, Welch averaging, and the relevant theorems. See [NOTES.md](NOTES.md) for project-specific conventions and measured anti-aliasing results.

## Examples and validation

- Open [the synthetic signal-analysis walkthrough](examples/signal_analysis_walkthrough.ipynb) from the repository root for standalone amplitude, PSD/power, indicator, downsampling, demodulation, spatial-resampling, and wavelength examples.
- Open [the measurement-handling walkthrough](examples/measurement_handling_walkthrough.ipynb) for independent time bases, units, provenance, derived signals, configured `Analysis`, and the envelope-order happy path.
- Reproduce the anti-aliasing comparison with:

  ```text
  python -m examples.spatial_antialiasing_study
  ```

- Run the deterministic test suite with:

  ```text
  python -m pytest -q
  ```

- The same command also executes package docstring examples through the installed import paths.
- GitHub Actions runs the test suite on supported Python versions and verifies
  that the wheel and source distribution build successfully.

## License

This project is available under the [MIT License](LICENSE).
