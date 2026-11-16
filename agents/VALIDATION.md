# Numerical test guide

Use deterministic synthetic signals and explicit tolerances. Prefer pure NumPy/SciPy + pytest. Keep tests small and fast.

## FFT

- Bin-centered sinusoid, known amplitude and frequency; verify `abs(Xf)` at the expected bin for rectangular/Hann windows, including one-sided scaling.
- DC and even-length Nyquist coefficients must **not** be doubled; odd-length has no Nyquist bin.
- Two-sided output retains expected bins and scaling.
- Phase should remain available through complex `Xf`. The window convention and FFT sign convention should be explicit.
- Distinguish amplitude spectrum from energy/PSD. Do not introduce a PSD claim.

## Averaged FFT

- Repeated identical blocks give the same mean magnitude as an individual block (when aligned with the selected window).
- Compare `return_spectrogram=False` vs `True`: frequencies and averages agree; `S.shape == (number_of_blocks, number_of_bins)` and `avg == S.mean(axis=0)`.
- Confirm correct behavior for overlap 0 and 0.5; incomplete last block omitted.
- For odd block sizes, confirm the sample hop is floored and returned block centers reflect the resulting effective overlap.
- Never overwrite time-domain `x` with FFT `Xf` during block iteration.

## PSD and power spectra

- Verify Parseval consistency against the quantity implied by the selected window and detrending. A rectangular, undetrended periodogram agrees exactly with the finite record's mean square; a tapered periodogram agrees with its window-power-normalized, window-weighted mean square and estimates the original signal power only under the corresponding signal assumptions.
- A bin-centered sinusoid with peak amplitude `A` has total AC power `A**2 / 2`; test rectangular and Hann windows.
- Cover DC, even-length Nyquist, one- and two-sided forms, explicit detrending, and seeded broadband noise.
- `averaged_psd` must equal the mean of individual block PSDs, use the documented floored hop, and omit incomplete trailing blocks.
- Never derive power by squaring `averaged_fft`; mean magnitude and mean squared magnitude are different operations.

## Coordinate extraction and indicators

- Coordinate ranges are half-open and coordinates must be finite and strictly increasing.
- Test population moments, RMS, crest factor, Pearson kurtosis, zero/constant-signal behavior, and optional energy with explicit sample spacing.
- Amplitude-band indicators remain magnitude-only. PSD-band power is `sum(PSD) * df`, with RMS as its square root.
- Reject nonuniform PSD frequency grids and amplitude/PSD kind mismatches in the analysis facade.

## Amplitude demodulation

- Use a deterministic amplitude-modulated carrier with known carrier amplitude, modulation depth and modulation frequency.
- Hilbert-envelope output retains amplitude units; square-law output has squared units. Test these conventions separately.
- Confirm the modulation survives a demodulation -> spatial resampling -> spatial spectrum workflow with the expected order and amplitude.
- Both filters are zero-phase and have endpoint transients. Exclude edge regions when validating steady-state amplitudes.

## Synthetic signals

- Use seeded noise and verify repeatability for identical seeds.
- For the tone mixer, test temporal-frequency and order components separately against known amplitudes and verify trapezoidal position for an analytically integrable speed profile.
- The bearing-like model is qualitative. Validate its declared speed range, synchronous output arrays and recovery of the configured fault order through the actual demodulation and spatial-resampling path; do not treat it as a calibrated mechanical model.

## Spatial resampling

- Constant speed: expected linear position progression and sample spacing `1 / resolution`.
- Variable positive speed: use analytically integrable speed where practical; compare integrated position and known constant-order sinusoid.
- A signal with known spatial order stays at that order through run-up/run-down (away from near-zero speed).
- Confirm output length and position coordinates from the AA wrapper and direct resampler, including a short input.
- Exact integrated grid boundaries are excluded consistently even when floating-point accumulation lands slightly above them; a genuine non-grid remainder retains its final valid grid point.
- Negative/zero speed not supported by current algorithm: don't invent position by clipping/flooring speed.

## Uniform resampling

- Confirm output rate and `ceil(N * up / down)` length for integer and rational factors.
- Test low-frequency amplitude retention, suppression of a component above the output Nyquist frequency, configurable axes, and short valid inputs.
- Keep uniform resampling tests separate from speed-to-position behavior.

## Explicit channel alignment

- `align_to_reference` retains only reference samples in the inclusive coordinate overlap, requires at least two retained samples, and never extrapolates.
- Validate different starts, rates, lengths, nonuniform coordinates, exact overlap endpoints, complex values, and unchanged inputs.
- Alignment is linear interpolation for slowly varying channels, not anti-aliased downsampling. Keep decimation tests in `resample_uniform`.
- `Analysis.align_pair` stores both the cropped reference and interpolated signal with matching coordinates and provenance. No other paired operation aligns implicitly.

## Anti-aliasing limitations

Current path: fixed temporal Butterworth filter at `max(speed) * resolution / 2` -> speed integration -> `np.interp` to an oversampled spatial grid -> `resample_poly` to the target grid. The temporal filter removes components that are outside the target band even at maximum speed, and the final FIR filter mitigates aliases during final decimation. Neither stage makes nonuniform-to-uniform reconstruction alias-free. Low-speed sections can map a component below the fixed temporal cutoff above the initial spatial-interpolation Nyquist frequency. A fixed safe cutoff based on *minimum* positive speed would remove useful bandwidth available at higher speed.

The reproducible comparison is `python -m examples.spatial_antialiasing_study`. The intended case uses a 4:1 speed ratio and a `resolution / 2.56` working band; the 10:1 cases are stress tests. When changing the filtering or interpolation, rerun both and distinguish passband attenuation, alias suppression, initial-interpolation limits and acquisition limits.

## File search

- Search nested directories for a specified extension.
- Verify alphabetical, case-insensitive sort by **filename**, not full path.
- Confirm return type is `Path`.

## DataFrame adapter

- Test explicit-column, DataFrame-index, and sample-rate coordinate modes separately; exactly one is required.
- Validate `ChannelSpec` units and metadata, automatic numeric-column selection, unchanged input frames, and nullable unknown units.
- Reject non-DataFrames and missing, duplicated, nonnumeric, nonfinite, or non-increasing selected data. The adapter supports one shared grid only and performs no file I/O.

## Measurement and analysis composition

- Preserve independent signal time/coordinate grids in one measurement.
- Paired numerical operations must reject unequal lengths, rates, starts, or coordinates rather than aligning silently; only `align_pair` may align explicitly.
- Derived signals append operation, input, output, and parameter provenance; duplicate names and unknown inputs are rejected.
- `AnalysisResult` accepts finite real scalars plus meaningful NaN values, rejects infinity, and requires a unit entry for every value. `None` means unknown and `"1"` means dimensionless.
- Signal and result names share one namespace. Stored results append ordinary provenance, while unnamed results remain structured but ephemeral.
- `Measurement` has no source or batch identifier, preserves uninterpreted metadata, and separate instances share no mutable state.
- Measurement-aware indicator methods always return `AnalysisResult`; standalone array functions remain plain numeric dictionaries for callers managing units themselves.
- Built-in amplitude and PSD results carry explicit spectrum-kind, normalization, and aggregation metadata. Full versus averaged spectra use `none` versus the appropriate magnitude- or power-density aggregation, and indicator dispatch rejects kind mismatches.
- Paired operations reject unequal coordinate-unit labels as well as unequal lengths, rates, starts, or coordinate values; no unit conversion is implicit.
- Validate the happy-path envelope-order chain against the configured synthetic fault order while retaining intermediate derived signals.
- Persistence and plotting are outside this layer and must not appear as hidden side effects.

## Examples and documentation

- Keep both committed notebooks valid, output-free, deterministic, and executable from the repository root.
- Execute both notebooks in memory during validation so generated plots and outputs are not committed.
- Run the anti-aliasing study after changes to filtering, interpolation, spatial-grid construction, or its documented reference cases.
- Keep README examples self-contained and executable with public functions.
- Verify editable and regular package imports through `signal_analyzer`; tests must not rely on the repository root containing importable modules.
- Keep file loading, source identity, multi-source iteration, persistence, scheduling, plotting, and GUI behavior outside the core.
