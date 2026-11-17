# Project status and backlog

## Completed foundation

- Deterministic pytest coverage for amplitude FFTs, PSD/power spectra, indicators, demodulation, uniform and spatial resampling, measurement composition, file discovery, and synthetic generators.
- Shared private validation for generic scalar/array contracts, with domain-specific validation kept in its owning module.
- Intentional FFT overlap rule: non-integer sample hops are floored and the resulting effective overlap is documented and tested.
- Deterministic half-open spatial output grids at exact and near-exact endpoint boundaries.
- Max-speed temporal anti-alias filter plus spatial oversample/polyphase wrapper, with reproducible 4:1 intended-envelope and 10:1 stress studies.
- Hilbert-envelope and square-law demodulation that composes before spatial resampling.
- Configurable tone/order synthesis and a qualitative variable-speed bearing-fault workflow signal.
- Explicit half-open coordinate extraction, signal statistics, amplitude-band indicators, and PSD band-power/RMS indicators.
- Generic rational polyphase resampling for uniformly sampled signals.
- Minimal in-memory `Signal`/`Measurement` data and provenance model plus a composed `Analysis` happy-path facade.
- Nullable units and structured named `AnalysisResult` artifacts with per-value units and provenance.
- Explicit overlap-only linear alignment for separately sampled channels, without implicit extrapolation or decimation.
- File-agnostic pandas DataFrame adapter for simple shared-grid measurements.
- Pip-installable `src/signal_analyzer` package with source identity and multi-source orchestration kept outside the core.
- Single-source installed version metadata, CI coverage for declared runtime
  dependency floors, and clean wheel/source-distribution smoke validation.
- Output-free, executable signal-analysis notebook covering the standalone numerical APIs and reciprocal wavelength interpretation.
- Output-free, executable measurement-handling notebook covering independent grids, provenance, derived signals, and the configured happy path.

## Open technical work

These are design questions, not automatic implementation tasks:

1. Design persistence only after the measurement schema has been exercised. Evaluate Parquet for tabular signals and indicators plus a manifest/provenance representation; compare array-oriented storage where appropriate.
2. Add plotting against stable signal/result objects after the measurement layer exists.
3. Define segmentation behavior for stationary, near-zero-speed, and reversing intervals only when that workflow is needed.
4. Compare a rigorous nonuniform reconstruction method only if the current anti-aliasing limits become material in real use.
5. Add a public wavelength-coordinate helper only if repeated callers need behavior beyond the notebook example.

## Only on explicit request

- Add additional notebooks, report generation, standardized application plots, publishing/release automation, CLI behavior, dashboards, databases, or similarity/clustering workflows.
- Integrate any external or private reader, format, metadata model, or application adapter. These should normally remain outside this repository.

## Completion criteria

- Keep the change narrow and avoid breaking public APIs without a documented reason.
- Add deterministic tests for every numerical behavior change.
- Run focused tests during implementation and `python -m pytest -q` before completion.
- Update conventions and limitations in `NOTES.md` when they change.
- Report changed files, verification results, and unresolved limitations succinctly.
