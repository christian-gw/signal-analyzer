# Numerical conventions and design notes

## Scope

This repository contains small, generic DSP building blocks, an in-memory measurement layer, deterministic synthetic examples, and a file-agnostic pandas DataFrame adapter. Private formats, application-specific metadata, and workplace adapters remain separate.

## Accepted conventions

### FFT and spectral averaging

- `fft_spectrum` returns complex coefficients. One-sided interior bins are doubled, while DC and even-length Nyquist bins are not.
- Coherent-gain correction divides by the window sum and estimates bin-centered sinusoidal amplitude. This is amplitude normalization, not PSD normalization.
- `averaged_fft` computes a mean magnitude spectrum, not a mean complex FFT or power spectrum. Incomplete trailing blocks are omitted.
- The integer hop is `int(block_size * (1 - overlap))`. A non-integer hop is floored, so effective overlap `1 - step / block_size` can be slightly larger than requested.
- Use `x` for the input signal and `Xf` for Fourier coefficients; do not overwrite the original signal during processing loops.
- `psd_spectrum` uses SciPy periodogram density normalization. `averaged_psd` averages block power densities with the same floored-hop convention as `averaged_fft`.
- PSD detrending is disabled by default so DC is preserved unless the caller explicitly requests detrending.
- `power_spectrum` is PSD multiplied by bin width `fs / N`; its values have signal-unit² per bin and their sum estimates mean square. It is not an amplitude spectrum.

### Range extraction and indicators

- Coordinate selections are half-open: `lower <= coordinate < upper`. Coordinates must be finite and strictly increasing.
- `signal_statistics` reports population moments, RMS, peaks, crest factor, skewness, Pearson kurtosis, and optional discrete energy when sample spacing is supplied.
- `amplitude_band_statistics` reports magnitude-only quantities and never infers RMS or power from amplitude values.
- `psd_band_statistics` requires a finite non-negative PSD on a uniformly spaced increasing coordinate. Band power is `sum(PSD) * bin_width`; band RMS is its square root.
- Arrays alone cannot prove spectral normalization. The `Analysis` facade classifies stored spectra through `Signal.kind` and rejects mismatched amplitude/PSD indicator dispatch.

### Demodulation

- `amplitude_demodulate` applies a temporal zero-phase Butterworth band-pass, Hilbert-envelope or square-law detection, and a zero-phase low-pass.
- Hilbert output retains the input amplitude unit. Square-law output has squared units. Demodulated mean removal is explicit and enabled by default.
- The result stays on the original time grid and can be passed to `speed_resample_aa`. This ordering is appropriate when the carrier band is specified in temporal frequency.
- Forward-backward filtering avoids causal delay but creates endpoint transients. Trim synchronized signal and speed endpoints before amplitude-sensitive processing.

### Speed-to-position resampling

- `speed_resample` requires synchronous, finite 1D signal and speed arrays, at least two samples, positive sampling rate and resolution, and strictly positive speed.
- Speed is integrated by the trapezoidal rule. Its unit may be revolutions/time, length/time, or another positive rate; resolution is samples per corresponding integrated unit.
- Output coordinates use a deterministic half-open grid: `0 <= pos < integrated final position`. Values within machine-precision tolerance of an exact grid count are snapped to prevent a spurious endpoint sample.
- `max_speed_aa_filter` defaults to an eighth-order zero-phase temporal Butterworth cutoff at `max(speed) * resolution / 2`. Filtering is skipped when that cutoff reaches acquisition Nyquist.
- `speed_resample_aa` applies the max-speed temporal filter, interpolates to a finer spatial grid, and uses FIR/polyphase decimation. This is partial protection, not rigorous anti-aliased nonuniform reconstruction.
- A useful grid-density check through the nominal max-speed cutoff is `oversample >= max(speed) / min(speed)`. For a working band ending at `resolution / 2.56`, the factor is approximately `0.78125 * max(speed) / min(speed)`; this remains a sizing heuristic, not a guarantee.

### Uniform resampling

- `resample_uniform` applies SciPy polyphase resampling on an explicitly selected uniform axis and returns the output sample rate `fs * up / down`.
- Uniform temporal downsampling remains separate from speed-to-position resampling because their coordinates and anti-aliasing questions differ.
- `align_to_reference` crops a reference grid to the inclusive coordinate overlap and linearly interpolates another signal onto it. It never extrapolates and is not an anti-aliased downsampler.

### Measurement and analysis composition

- `Signal` stores finite 1D values on either a uniform grid (`sample_rate` and `start`) or an explicit strictly increasing coordinate. Signals in one `Measurement` keep independent time bases. `None` denotes an unknown unit and `"1"` an explicitly dimensionless unit.
- `AnalysisResult` stores named scalar values with an exactly matching per-value unit map. NaN is allowed for meaningful undefined statistics; infinities are rejected.
- `Measurement` is an unnamed in-memory container for original and derived signals, named scalar results, metadata, and plain `OperationRecord` provenance. Signal and result names share one namespace. File, job, GUI, and persistence identity remain external.
- `Analysis` composes the standalone functions with defaults for spatial resolution and oversampling. The documented `resolution / 2.56` practical working boundary remains a user-side sizing heuristic rather than an API property. `Analysis` adds derived signals and provenance rather than reimplementing numerical algorithms.
- Standalone indicator functions return numeric dictionaries so callers may manage units themselves. Measurement-aware `Analysis` indicator methods always return `AnalysisResult`; `output_name` additionally stores the result and provenance.
- Generated amplitude and PSD spectra carry explicit kind, normalization, and aggregation metadata. Operations that require ordinary signals reject stored spectra.
- Full and averaged amplitude spectra use coherent-gain normalization; their aggregation is respectively `none` or `mean_magnitude`. Full and averaged PSDs use density normalization; their aggregation is respectively `none` or `mean_power_density`.
- `Analysis.align_pair` performs explicit overlap-only linear interpolation and retains both aligned outputs. Other paired operations still require matching grids and coordinate-unit labels and never align implicitly.
- Unit strings are propagated and compared but are not inferred, parsed, or converted dimensionally. Derived units remain unknown when a required input unit is unknown.
- The envelope-order happy path retains demodulated, trimmed, spatial, and spectral intermediate signals for diagnosis.

### Readers, packaging, synthetic signals, and file discovery

- `measurement_from_dataframe` converts one in-memory pandas DataFrame to one shared-grid `Measurement` without file I/O. Exactly one of an explicit coordinate column, the DataFrame index, or a sample rate is required. `ChannelSpec` supplies optional units and metadata; units are never guessed.
- File-format loading, source identity, multi-source iteration, and multi-file measurement assembly belong to application code. Independent grids can be added directly as separate `Signal` objects.
- The installable distribution is `signal-analyzer`; imports use the `signal_analyzer` package under the `src` layout.
- `synthesize_tone_mix` creates additive cosine components on temporal-frequency and integrated-position/order coordinates. Speed may be scalar, an array, or a callable; seeded Gaussian noise is deterministic.
- `simulate_bearing_fault` combines smooth variable speed, shaft orders 1–5, fixed temporal tones, and fault-order impacts exciting a decaying resonance. It is qualitative and generic, not a calibrated machine model.
- Both generators return synchronous NumPy arrays `(t, speed, position, x)`.
- `find_files` uses `Path.rglob`, returns `Path` objects, and sorts case-insensitively by filename rather than full path.

## Synthetic anti-aliasing study

Reproduce the study with `python -m examples.spatial_antialiasing_study`. Its dense temporal sampling, 16× spatial interpolation, and polyphase decimation provide a numerical reference for the chosen synthetic signals, not a proof of ideal reconstruction.

- At constant 8 units/s and target spatial Nyquist 32 cycles/unit, a 36 cycles/unit tone aliased at amplitude 1.000 with direct interpolation, 0.0315 with spatial-only 4× processing, and 0.0039 with the max-speed filter plus 4× processing.
- At the practical `resolution / 2.56` boundary, a constant-speed 25 cycles/unit tone retained amplitude 0.982 with the temporal filter plus 4× processing, versus 1.000 for spatial-only processing.
- In the intended 4:1 speed envelope, default 4× processing differed from a 16× spatial grid by 0.31% RMS at mid-band. Relative error against the dense reference was 0.31% at mid-band and 2.88% at the practical band edge; the latter is dominated by temporal-filter roll-off.
- A 300 Hz guard-band tone in the 4:1 case had 26.8% relative error against the unfiltered dense reference because it lies close to the 320 Hz temporal cutoff. This is outside the `/2.56` working band and illustrates the transition region.
- In the 10:1 stress case, a 240 Hz tone below the max-speed cutoff produced 13.4% relative RMS error with 4× processing and 2.0% with 8× processing. The fixed temporal filter cannot remove every component that becomes spatially undersampled at low speed.
- A 480 Hz tone above the max-speed cutoff was strongly suppressed: with 4× processing, output RMS fell from 0.1001 for spatial-only processing to 0.0001 with the temporal filter.

## Deferred generic integration ideas

- External ingestion may provide formats or multi-file measurements beyond the deliberately limited shared-grid DataFrame adapter.
- Persistence should follow, not define, the in-memory schema. Parquet plus a manifest is a candidate for tabular signals, indicators, and metadata; array-oriented formats remain alternatives for large or highly nested signal data.
- Intermediate time-, position-, demodulated-, and spectral-domain results may be useful for diagnosis and reproducible plotting, but this repository does not prescribe a workflow framework.

## Open technical decisions

- **Stationary or reversing motion:** position mapping becomes degenerate. Segment or reject these intervals explicitly; never floor speed to invent displacement.
- **Nonuniform anti-aliasing:** compare a direct reconstruction method only if the current max-speed filter and oversampling limits become material in real use.
- **Wavelength API:** the notebook demonstrates reciprocal positive-frequency coordinates, but there is no public helper. Any future helper must omit zero, state units, reverse ordering deliberately, and acknowledge nonuniform wavelength spacing.
- **Ownership boundary:** keep public numerical methods, the in-memory model, the generic DataFrame adapter, synthetic signals, and generic examples here. File readers, private data, external identifiers, and reports remain outside the repository.
