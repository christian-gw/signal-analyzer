# Spectral analysis: amplitudes, power, PSD, windows, and averaging

This is a short reference for the spectral quantities used in this repository. The central rule is simple: **an amplitude spectrum, power spectrum, and power spectral density are different objects**. Their values, units, averaging rules, and valid indicators are not interchangeable.

## 1. From samples to Fourier coefficients

For `N` uniformly sampled values `x[n]`, NumPy uses the unnormalized forward discrete Fourier transform (DFT):

\[
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}.
\]

The FFT is only a fast algorithm for calculating this DFT. The frequency spacing is

\[
\Delta f = \frac{f_s}{N}.
\]

For temporal data, `fs` is samples per time unit and the Fourier coordinate is cycles per time unit, normally Hz. For spatial data, `fs` is samples per integrated spatial unit and the coordinate is cycles per that unit—for revolutions, this is order.

Increasing `N` by observing a longer record improves actual frequency resolution. Zero-padding only evaluates the same finite-record spectrum on a denser coordinate grid; it does not separate components that the original record could not resolve.

## 2. The spectra implemented here

### Complex amplitude spectrum

With its default `window_correction=True`, `fft_spectrum` returns complex coefficients with coherent-gain amplitude normalization. For a bin-centered real sinusoid

\[
x(t) = A\cos(2\pi f_0t + \phi),
\]

the corresponding one-sided bin has magnitude approximately `A`, the sinusoid's **peak amplitude**, and retains phase information. Interior one-sided bins are doubled; DC and even-length Nyquist bins are not. In a two-sided spectrum, the positive- and negative-frequency coefficients each carry half of a real sinusoid's peak amplitude. Disabling window correction leaves the coherent attenuation of a non-rectangular window in the result.

This is appropriate for identifying deterministic lines and their amplitudes. It is not a PSD and its squared values are not automatically power-per-bin values, particularly after window coherent-gain correction.

### Averaged magnitude spectrum

`averaged_fft` calculates an amplitude spectrum for each block and returns

\[
\operatorname{mean}(|X_k|).
\]

This stabilizes displayed line amplitudes but remains a **linear-magnitude average**. It is not Welch power averaging:

\[
\operatorname{mean}(|X_k|)^2 \neq \operatorname{mean}(|X_k|^2).
\]

Consequently, band power or RMS must not be reconstructed by squaring an `averaged_fft` result.

### Power spectral density

`psd_spectrum` returns a periodogram with window-power normalization. Its units are

\[
\frac{\text{signal unit}^2}{\text{cycles per coordinate unit}}.
\]

For time in seconds this is commonly `unit²/Hz`. Integrating—or, for uniform FFT bins, summing with the bin width—estimates mean square:

\[
\overline{x^2} \approx \sum_k P_{xx}[k]\,\Delta f.
\]

Therefore

\[
x_\mathrm{RMS} \approx \sqrt{\sum_k P_{xx}[k]\,\Delta f}.
\]

`psd_spectrum` preserves DC by default because `detrend=False`. Removing the mean or another trend is a deliberate analysis decision.

With a non-rectangular window, this equality is exact for the window-weighted record after window-power normalization. It estimates the unwindowed record's mean square when the signal power is reasonably stationary across the record; transients concentrated near a tapered endpoint can differ. Direct time-domain RMS remains the reference for a particular finite record.

### Power per bin

`power_spectrum` returns

\[
P_\mathrm{bin}[k] = P_{xx}[k]\,\Delta f.
\]

Its units are `signal unit² per bin`, and its sum estimates mean square. Unlike PSD, power-per-bin changes when `N` and therefore `Δf` change. PSD is usually preferable when comparing records with different lengths or resolutions.

### Averaged PSD

`averaged_psd` calculates a PSD for each complete block and averages the power densities. It uses the same floored integer-hop convention as `averaged_fft`, but its statistical meaning is different because it averages power rather than magnitude.

## 3. RMS, power, and energy

For a sampled signal:

\[
x_\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_n x[n]^2},
\qquad
\text{mean square} = x_\mathrm{RMS}^2.
\]

With sample spacing `Δt`, finite-record energy is commonly approximated as

\[
E = \sum_n x[n]^2\,\Delta t.
\]

Energy therefore has an additional coordinate unit. PSD integration gives mean square or average power, not finite-record energy; energy additionally depends on duration. In the spatial domain the same mathematics yields mean square per spatial record, but its physical interpretation is spatial rather than temporal.

`signal_statistics` calculates RMS directly from samples and only reports energy when sample spacing is supplied. `psd_band_statistics` calculates band power and band RMS from PSD. `amplitude_band_statistics` deliberately reports no power or RMS.

## 4. Why windows are needed

The DFT treats the finite record as one period of a repeated signal. If the record endpoints do not join continuously, energy spreads into neighboring bins: **spectral leakage**.

A window tapers the record to reduce this discontinuity. Every window trades among:

- Main-lobe width: ability to separate nearby components.
- Side-lobe level: leakage from a strong component into weak nearby components.
- Scalloping loss: amplitude error when a tone lies between bins.
- Equivalent noise bandwidth (ENBW): noise admitted by one spectral estimate.
- Coherent gain: amplitude reduction applied to a bin-centered sinusoid.

Approximate values for common periodic FFT windows are:

| Window | Coherent gain | ENBW [bins] | Main lobe, zero-to-zero [bins] | Typical first/highest side lobe | Good default use |
|---|---:|---:|---:|---:|---|
| Rectangular | 1.00 | 1.00 | 2 | −13 dB | Exact bin-centered lines, maximum resolution |
| Hann | 0.50 | 1.50 | 4 | −31 dB | General vibration and spectral averaging |
| Hamming | 0.54 | 1.36 | 4 | about −43 dB | Reduced nearest side lobe |
| Blackman | 0.42 | 1.73 | 6 | about −58 dB | Strong leakage suppression with wider peaks |
| Flat top | about 0.216 | about 3.77 | about 10 | about −93 dB | Accurate off-bin line amplitude, poor resolution |

Values vary slightly with window definition and finite length. A Hann window is a good general default. Rectangular is best only when leakage is controlled by coherent sampling or is otherwise acceptable. Flat top is useful when line-amplitude accuracy matters more than separating nearby frequencies.

### Coherent gain versus window power

For window samples `w[n]`, coherent gain is

\[
CG = \frac{1}{N}\sum_n w[n].
\]

Amplitude normalization divides by the window sum. PSD normalization instead uses window power, proportional to

\[
\sum_n w[n]^2.
\]

This is why one normalization cannot be silently reused for the other.

The equivalent noise bandwidth in FFT-bin units is

\[
ENBW_\mathrm{bins}
= \frac{N\sum_n w[n]^2}{\left(\sum_n w[n]\right)^2}.
\]

## 5. Welch's method

Welch's method is an estimator, not a theorem. It:

1. Divides a record into blocks.
2. Windows each block.
3. Calculates a periodogram for each block.
4. Averages the block power densities.

The result has lower random variance than one full-record periodogram, at the cost of shorter blocks and therefore coarser frequency resolution. Overlap recovers more use from tapered endpoints, but overlapping blocks are correlated; increasingly large overlap eventually gives diminishing statistical benefit.

`averaged_psd` is the project's Welch-style power-density average. `averaged_fft` is not Welch PSD because it averages magnitudes instead of powers. Confidence intervals and statistical degrees of freedom are not currently calculated.

## 6. Theorems behind the normalizations

### Parseval/Plancherel

For NumPy's DFT convention, Parseval's relation is

\[
\sum_n |x[n]|^2 = \frac{1}{N}\sum_k |X[k]|^2.
\]

It connects time-domain mean square with total spectral power and is the central validation check for `psd_spectrum` and `power_spectrum`. With a taper, apply the relation to the windowed data and its window-power normalization as described above. One-sided spectra must preserve the corresponding total through correct doubling of interior positive-frequency power.

### Wiener–Khinchin

The Wiener–Khinchin theorem states that a stationary process's PSD is the Fourier transform of its autocorrelation. A finite-record periodogram is an estimate of that underlying PSD, which explains why random-signal PSDs fluctuate and why averaging improves stability.

### Cauchy–Schwarz and window ENBW

Applying the Cauchy–Schwarz inequality to the window gives

\[
\left(\sum_n w[n]\right)^2 \leq N\sum_n w[n]^2.
\]

Therefore `ENBW_bins >= 1`; the rectangular window reaches equality. Any taper that suppresses leakage necessarily broadens effective noise bandwidth relative to a rectangular window. This does not make rectangular universally best—it states one part of the unavoidable leakage/resolution/noise tradeoff.

The relevant Schwarz result for spectral windows is the Cauchy–Schwarz inequality. Schwarz's theorem about exchanging mixed partial derivatives is a different result and has no direct role here.

## 7. Sampling, Nyquist, and aliasing

For uniform sampling, frequencies above `fs / 2` cannot be uniquely represented and fold into the sampled band. A digital filter applied after acquisition cannot undo aliases already created during acquisition.

`resample_uniform` applies polyphase filtering before uniform-grid decimation. Speed-to-position resampling is harder because temporal frequency maps to spatial frequency as speed changes. `speed_resample_aa` combines a max-speed temporal filter with spatial oversampling and polyphase decimation, but remains a partial safeguard rather than rigorous nonuniform reconstruction.

## 8. Related spectral objects

These are useful distinctions even though the toolkit does not yet expose dedicated helpers for all of them:

- **Amplitude spectral density (ASD)** is `sqrt(PSD)`, with units such as `unit/sqrt(Hz)`. Square it before integrating; ASD values themselves do not add as power.
- **Energy spectral density** describes the frequency distribution of a finite-energy signal. It is related to `|X(f)|²` with transform-dependent scaling, rather than the long-term average-power interpretation of PSD.
- **Cross spectral density** retains shared frequency-dependent phase and power between two signals. **Coherence** normalizes it to show the strength of their linear frequency-domain relationship.
- A **spectrogram** or short-time Fourier transform keeps block position as well as frequency. The per-block arrays returned by `averaged_fft(..., return_spectrogram=True)` and `averaged_psd(..., return_spectrogram=True)` provide the underlying blockwise views, but no plotting abstraction is imposed.

## 9. Practical selection guide

- Deterministic line amplitude and phase: `fft_spectrum`.
- Stable displayed line amplitudes across blocks: `averaged_fft`.
- Broadband noise level or comparable density: `psd_spectrum`.
- Lower-variance broadband estimate: `averaged_psd`.
- Mean-square contribution per FFT bin: `power_spectrum`.
- Overall or time-zone RMS: `signal_statistics` on the selected samples.
- Band peak or mean amplitude: `amplitude_band_statistics`.
- Band power or band RMS: `psd_band_statistics`.

Choose block size from the required resolution `Δf = fs / block_size`, then choose a window from the leakage and amplitude-accuracy requirements. Averaging reduces variance; it does not restore resolution lost through short blocks.

The standalone band-statistics helpers require strictly increasing coordinates. Two-sided FFT and periodogram outputs use FFT ordering—zero, positive frequencies, then negative frequencies—and must be reordered before being passed to those helpers. The `Analysis` facade creates one-sided spectra and therefore avoids this extra step.

## 10. Common mistakes

- Squaring a mean magnitude spectrum and calling it average power.
- Summing PSD bins without multiplying by bin width.
- Comparing power-per-bin spectra made with different frequency resolutions.
- Treating a window-corrected amplitude spectrum as Parseval-normalized power.
- Forgetting that detrending removes DC or low-frequency content.
- Reading zero-padding as increased physical resolution.
- Reporting wavelength `1/f` at `f=0`, or forgetting that wavelength bins are nonuniform and reverse frequency ordering.
- Calling a speed-resampled spectrum alias-free without testing the relevant speed range and bandwidth.

The deterministic checks in `tests/test_fft_tools.py` and `tests/test_indicators.py` encode these conventions for the implemented functions.
