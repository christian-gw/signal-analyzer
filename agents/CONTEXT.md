# Project context and boundaries

## Current public API

- `signal_analyzer.fft_tools`: amplitude, averaged-magnitude, PSD-density, averaged-PSD, and power-per-bin spectra.
- `signal_analyzer.demodulation.amplitude_demodulate`: temporal zero-phase band-pass, Hilbert-envelope or square-law detection, and zero-phase low-pass.
- `signal_analyzer.indicators`: half-open coordinate selection plus signal, amplitude-band, and PSD-band scalar indicators.
- `signal_analyzer.sampling`: rational uniform polyphase resampling plus explicit overlap-only linear alignment.
- `signal_analyzer.synthetic_signals`: configurable tone mixtures and a qualitative variable-speed bearing-fault workflow signal.
- `signal_analyzer.resampling`: speed integration, max-speed temporal filtering, spatial interpolation, and polyphase decimation.
- `signal_analyzer.measurement`: nullable-unit `Signal`, structured `AnalysisResult`, unnamed `Measurement`, and `OperationRecord` artifacts.
- `signal_analyzer.analysis.Analysis`: measurement-aware composition, unit-bearing indicators, explicit alignment, and the envelope-order happy path.
- `signal_analyzer.tabular`: file-agnostic pandas DataFrame conversion through `measurement_from_dataframe` and `ChannelSpec`.
- `signal_analyzer.file_utils.find_files`: recursive `Path.rglob` discovery sorted case-insensitively by filename.

## Numerical boundaries

- `fft_spectrum` uses coherent-gain amplitude normalization by default. For a bin-centered real sinusoid, a one-sided magnitude estimates peak amplitude; two-sided output splits that amplitude across positive and negative frequency bins. `averaged_fft` averages magnitudes, not power or complex coefficients.
- PSD outputs use window-power density normalization; power-per-bin values are PSD multiplied by frequency-bin width. Only PSD-based indicators calculate spectral band power or RMS.
- Spectrum classification uses `Signal.kind`; normalization and aggregation metadata are attached by the built-in `Analysis` methods. Arrays from arbitrary external functions cannot prove their own normalization.
- Temporal sampling rate is samples per time unit; speed must use that same time unit in its denominator because units are not converted. Spatial sampling rate is samples per integrated speed unit. The corresponding FFT axes are cycles per those units.
- Spatial resampling requires synchronous finite arrays and strictly positive speed. Stationary or reversing sections are not supported and must not be hidden by speed flooring.
- Position grids are half-open and deterministic at floating-point endpoint boundaries.
- Anti-aliasing is partial: a fixed max-speed temporal filter and spatial oversample/polyphase stage cannot guarantee alias-free variable-speed reconstruction.
- Demodulation uses forward-backward filters. It has no causal delay but does have endpoint transients; trim synchronized signal and speed arrays when amplitude matters.
- The bearing-fault generator is a qualitative workflow example, not a calibrated mechanical model.
- Measurements preserve independent signal grids. Stored signal arrays are copied and read-only; derived registration validates provenance before mutation, and derived signals copy source metadata. `align_pair` aligns only when explicitly requested; other paired operations require equal coordinates, sample rates, and coordinate-unit labels.
- Unknown units are `None`, dimensionless units are `"1"`, and the core does not infer or convert units.

## Intended use

Use the functions directly from small scripts, or wrap named arrays in the optional in-memory measurement layer. Typical work includes temporal or PSD spectra, coordinate-range indicators, uniform downsampling, demodulation, speed-to-position transformation, and spatial amplitude spectra.

- [The signal-analysis walkthrough](../examples/signal_analysis_walkthrough.ipynb) exercises standalone amplitude, PSD/power, indicator, uniform-resampling, demodulation, and spatial-analysis functions.
- [The measurement-handling walkthrough](../examples/measurement_handling_walkthrough.ipynb) demonstrates independent grids, provenance, derived signals, configured analysis, and retained intermediates.
- `python -m examples.spatial_antialiasing_study` reproduces the constant- and variable-speed anti-aliasing comparison.
- [`SPECTRAL_ANALYSIS.md`](../SPECTRAL_ANALYSIS.md) summarizes amplitude, power, PSD, window, Welch, and normalization conventions used by the toolkit.

## Deferred integration

- File formats, source identity, iteration over multiple sources, and multi-file measurement assembly remain application responsibilities.
- Persistence and plotting are intentionally absent from the in-memory measurement and analysis layer.
- Additional report generation, application-specific plotting, private readers, and application adapters remain outside the current toolkit.

## Design preference

Prefer plain Python and NumPy/SciPy, informative `ValueError`s, deterministic synthetic tests, and directly runnable examples. Add pandas only where it clearly improves a requested higher-level container. Avoid abstractions that do not simplify an existing workflow.
