# Codex instructions — DSP scratchpad

This repository is a small, personal, generic DSP toolkit for exploratory vibration, temporal-frequency, order, spatial, and envelope analysis. It is not a framework, production application, or application-specific analysis project.

Before substantial changes, read `README.md`, `NOTES.md`, `agents/CONTEXT.md`, `agents/TASKS.md`, and the relevant sections of `agents/VALIDATION.md`. Read `SPECTRAL_ANALYSIS.md` before changing spectral normalization, windowing, PSD, power, or related indicators.

## Working rules

- Prefer small, independent functions usable directly from scripts. Keep NumPy inputs and outputs unless a higher-level container has a demonstrated need for pandas.
- Do not add a plugin system, CLI, service, dashboard, framework, unnecessary class hierarchy, or mandatory configuration without an explicit request.
- Preserve public names and signatures unless a change is justified, tested, and documented. Public modules live under `src/signal_analyzer/`: `fft_tools.py`, `demodulation.py`, `indicators.py`, `sampling.py`, `synthetic_signals.py`, `resampling.py`, `measurement.py`, `analysis.py`, `tabular.py`, and `file_utils.py`; `_validation.py` is private.
- Use `x` for an input signal and `Xf` for FFT coefficients. Never overwrite the original signal in processing loops.
- State units, normalization, and coordinate systems explicitly. Speed may be revolutions/time, length/time, or another positive rate; resolution is samples per corresponding integrated unit.
- For every numerical behavior change, add a small deterministic synthetic test with a known expectation. Prefer pytest and NumPy/SciPy.
- Do not replace amplitude or mean-magnitude behavior with power/PSD behavior without an explicit API and documentation change.
- Keep amplitude, power-per-bin, and PSD-density semantics explicit. Do not calculate band power or band RMS from an averaged amplitude spectrum.
- Do not claim rigorous anti-aliasing. The current path combines a fixed max-speed temporal filter, spatial interpolation, and polyphase decimation; it remains a partial safeguard. See `agents/VALIDATION.md`.
- Do not silently clamp speed near zero. Stationary and reversing intervals require explicit segmentation or rejection because clamping invents displacement.
- Favor narrow patches, readable functions, short docstrings, and executable examples. Update documentation when behavior or decisions change, or when documentation work is explicitly requested.
- Avoid placing all channels on one high-frequency time base unless the calling workflow explicitly requires it.
- Keep persistence and plotting separate from the numerical and in-memory measurement layers unless explicitly requested.

## Data and IP boundaries

- Use only public/generic methods, deterministic synthetic data, and simple examples.
- Do not import, reproduce, or infer private file formats, loaders, identifiers, metadata, code, or measurements.
- Keep file readers, private adapters, reports, and organization-specific terminology outside this repository. The generic shared-grid pandas DataFrame adapter must not acquire file-format or application-specific conventions.

## Session protocol

1. Follow the user’s explicit task. If none is specified, choose one open item from `agents/TASKS.md`.
2. Inspect the relevant code and notes, then establish baseline behavior with the existing tests or a focused smoke test.
3. Make the smallest coherent implementation and test changes. Avoid unrelated refactoring.
4. Update `README.md`, `NOTES.md`, and agent references only where the public behavior, decisions, project inventory, or explicit documentation request requires it.
5. Run focused tests during implementation and the full suite before reporting completion.
6. Report changed files, verification results, and open limitations without implying guarantees that were not tested.
