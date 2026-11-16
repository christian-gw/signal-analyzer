"""Composed happy-path analysis operations for a Measurement."""

import numpy as np

from ._validation import (
    require_nonnegative_finite_scalar,
    require_positive_finite_scalar,
    require_positive_integer,
)
from .demodulation import amplitude_demodulate
from .fft_tools import (
    averaged_fft,
    averaged_psd,
    fft_spectrum,
    psd_spectrum,
)
from .indicators import (
    amplitude_band_statistics as amplitude_statistics_in_band,
    coordinate_range_mask,
    psd_band_statistics as psd_statistics_in_band,
    signal_statistics as calculate_signal_statistics,
)
from .measurement import AnalysisResult, Measurement, Signal
from .resampling import speed_resample_aa
from .sampling import align_to_reference, resample_uniform


def _squared_unit(unit):
    if unit is None or unit == "1":
        return unit
    return f"({unit})^2"


def _product_unit(first, second):
    if first is None or second is None:
        return None
    if first == "1":
        return second
    if second == "1":
        return first
    return f"{first}*{second}"


def _frequency_unit(coordinate_unit):
    if coordinate_unit is None:
        return None
    if coordinate_unit in {"s", "sec", "second", "seconds"}:
        return "Hz"
    return f"cycles/{coordinate_unit}"


def _density_unit(value_unit, frequency_unit):
    squared = _squared_unit(value_unit)
    if squared is None or frequency_unit is None:
        return None
    if frequency_unit == "1":
        return squared
    return f"{squared}/{frequency_unit}"


class Analysis:
    """Configured orchestration around standalone numerical functions.

    Methods add derived signals and provenance to ``measurement``. Numerical
    work remains in the public standalone functions imported above.
    """

    def __init__(
        self,
        measurement,
        resolution=64,
        oversample=4,
    ):
        if not isinstance(measurement, Measurement):
            raise ValueError("measurement must be a Measurement")
        require_positive_finite_scalar(resolution, "resolution")
        require_positive_integer(oversample, "oversample")
        self.measurement = measurement
        self.resolution = float(resolution)
        self.oversample = int(oversample)

    def _signal(self, name):
        return self.measurement.signal(name)

    @staticmethod
    def _require_signal_kind(signal):
        if signal.kind != "signal":
            raise ValueError("operation requires kind='signal'")
        return signal

    @staticmethod
    def _sample_rate(signal):
        sample_rate = signal.inferred_sample_rate()
        if sample_rate is None:
            raise ValueError("operation requires a uniformly sampled signal")
        return sample_rate

    @staticmethod
    def _synchronized(first, second):
        if len(first.values) != len(second.values):
            raise ValueError("paired signals must have equal length")
        if not np.allclose(
            first.coordinates, second.coordinates, rtol=0, atol=1e-12
        ):
            raise ValueError("paired signals must use synchronized coordinates")
        first_rate = Analysis._sample_rate(first)
        second_rate = Analysis._sample_rate(second)
        if not np.isclose(first_rate, second_rate, rtol=1e-12, atol=0):
            raise ValueError("paired signals must have equal sample rates")
        if first.coordinate_unit != second.coordinate_unit:
            raise ValueError("paired signals must have matching coordinate units")
        return first_rate

    @staticmethod
    def _same_grid_signal(source, values, value_unit=None, kind=None):
        kwargs = {
            "values": values,
            "value_unit": source.value_unit if value_unit is None else value_unit,
            "coordinate_unit": source.coordinate_unit,
            "kind": source.kind if kind is None else kind,
            "metadata": source.metadata,
        }
        if source.sample_rate is not None:
            kwargs.update(sample_rate=source.sample_rate, start=source.start)
        else:
            kwargs.update(coordinate=source.coordinate)
        return Signal(**kwargs)

    def _analysis_result(
        self,
        kind,
        values,
        units,
        signal_name,
        lower,
        upper,
        output_name,
    ):
        parameters = {"lower": lower, "upper": upper}
        result = AnalysisResult(
            kind,
            values,
            units,
            metadata={"source": signal_name, **parameters},
        )
        if output_name is not None:
            self.measurement.add_derived_result(
                output_name,
                result,
                kind,
                (signal_name,),
                parameters,
            )
        return result

    def demodulate(
        self,
        signal_name,
        output_name,
        band,
        lowpass,
        detector="hilbert",
        filter_order=4,
        remove_mean=True,
    ):
        source = self._require_signal_kind(self._signal(signal_name))
        fs = self._sample_rate(source)
        values = amplitude_demodulate(
            source.values,
            fs,
            band,
            lowpass,
            detector=detector,
            filter_order=filter_order,
            remove_mean=remove_mean,
        )
        value_unit = (
            source.value_unit if detector == "hilbert"
            else _squared_unit(source.value_unit)
        )
        result = self._same_grid_signal(
            source, values, value_unit=value_unit, kind="signal"
        )
        return self.measurement.add_derived_signal(
            output_name,
            result,
            "amplitude_demodulate",
            (signal_name,),
            {
                "band": list(band),
                "lowpass": lowpass,
                "detector": detector,
                "filter_order": filter_order,
                "remove_mean": remove_mean,
            },
        )

    def downsample(self, signal_name, output_name, down, up=1):
        source = self._require_signal_kind(self._signal(signal_name))
        fs = self._sample_rate(source)
        values, fs_out = resample_uniform(
            source.values, fs, up=up, down=down
        )
        result = Signal(
            values,
            sample_rate=fs_out,
            start=source.coordinates[0],
            value_unit=source.value_unit,
            coordinate_unit=source.coordinate_unit,
            kind=source.kind,
        )
        return self.measurement.add_derived_signal(
            output_name,
            result,
            "resample_uniform",
            (signal_name,),
            {"up": up, "down": down},
        )

    def align_pair(
        self,
        reference_name,
        signal_name,
        reference_output_name,
        signal_output_name,
    ):
        """Align two signals explicitly on the reference grid overlap.

        The reference is cropped to samples inside the shared coordinate
        range. The other signal is linearly interpolated to those coordinates.
        No extrapolation or anti-alias filtering is performed.
        """
        reference = self._require_signal_kind(self._signal(reference_name))
        source = self._require_signal_kind(self._signal(signal_name))
        if reference.coordinate_unit != source.coordinate_unit:
            raise ValueError("paired signals must have matching coordinate units")
        output_names = (reference_output_name, signal_output_name)
        if output_names[0] == output_names[1]:
            raise ValueError("aligned output names must be different")
        for output_name in output_names:
            if output_name in self.measurement.signals or output_name in self.measurement.results:
                raise ValueError(f"name already exists: {output_name}")

        coordinate, reference_values, aligned_values = align_to_reference(
            reference.values,
            reference.coordinates,
            source.values,
            source.coordinates,
        )
        aligned_reference = Signal(
            reference_values,
            coordinate=coordinate,
            value_unit=reference.value_unit,
            coordinate_unit=reference.coordinate_unit,
            kind=reference.kind,
            metadata=reference.metadata,
        )
        aligned_source = Signal(
            aligned_values,
            coordinate=coordinate,
            value_unit=source.value_unit,
            coordinate_unit=source.coordinate_unit,
            kind=source.kind,
            metadata=source.metadata,
        )
        parameters = {
            "reference": reference_name,
            "method": "linear",
            "bounds": "coordinate_overlap",
        }
        self.measurement.add_derived_signal(
            reference_output_name,
            aligned_reference,
            "crop_to_coordinate_overlap",
            (reference_name, signal_name),
            parameters,
        )
        self.measurement.add_derived_signal(
            signal_output_name,
            aligned_source,
            "align_to_reference",
            (reference_name, signal_name),
            parameters,
        )
        return aligned_reference, aligned_source

    def spatial_resample(
        self,
        signal_name,
        speed_name,
        output_name,
        resolution=None,
        oversample=None,
        position_unit="integrated unit",
    ):
        source = self._require_signal_kind(self._signal(signal_name))
        speed = self._require_signal_kind(self._signal(speed_name))
        fs = self._synchronized(source, speed)
        resolution = self.resolution if resolution is None else resolution
        oversample = self.oversample if oversample is None else oversample
        pos, values = speed_resample_aa(
            source.values,
            speed.values,
            fs,
            resolution=resolution,
            oversample=oversample,
        )
        result = Signal(
            values,
            coordinate=pos,
            value_unit=source.value_unit,
            coordinate_unit=position_unit,
            kind="signal",
        )
        return self.measurement.add_derived_signal(
            output_name,
            result,
            "speed_resample_aa",
            (signal_name, speed_name),
            {"resolution": resolution, "oversample": oversample},
        )

    def amplitude_spectrum(
        self,
        signal_name,
        output_name,
        block_size=None,
        overlap=0.5,
        window="hann",
    ):
        source = self._require_signal_kind(self._signal(signal_name))
        fs = self._sample_rate(source)
        if block_size is None:
            coordinate, Xf = fft_spectrum(source.values, fs=fs, window=window)
            values = np.abs(Xf)
            operation = "fft_spectrum"
        else:
            coordinate, values = averaged_fft(
                source.values,
                block_size,
                overlap=overlap,
                fs=fs,
                window=window,
            )
            operation = "averaged_fft"
        result = Signal(
            values,
            coordinate=coordinate,
            value_unit=source.value_unit,
            coordinate_unit=_frequency_unit(source.coordinate_unit),
            kind="amplitude_spectrum",
            metadata={
                "source_value_unit": source.value_unit,
                "normalization": "coherent_gain",
                "aggregation": (
                    "none" if block_size is None else "mean_magnitude"
                ),
            },
        )
        return self.measurement.add_derived_signal(
            output_name,
            result,
            operation,
            (signal_name,),
            {"block_size": block_size, "overlap": overlap, "window": window},
        )

    def psd(
        self,
        signal_name,
        output_name,
        block_size=None,
        overlap=0.5,
        window="hann",
        detrend=False,
    ):
        source = self._require_signal_kind(self._signal(signal_name))
        fs = self._sample_rate(source)
        if block_size is None:
            coordinate, values = psd_spectrum(
                source.values,
                fs=fs,
                window=window,
                detrend=detrend,
            )
            operation = "psd_spectrum"
        else:
            coordinate, values = averaged_psd(
                source.values,
                block_size,
                overlap=overlap,
                fs=fs,
                window=window,
                detrend=detrend,
            )
            operation = "averaged_psd"
        frequency_unit = _frequency_unit(source.coordinate_unit)
        result = Signal(
            values,
            coordinate=coordinate,
            value_unit=_density_unit(source.value_unit, frequency_unit),
            coordinate_unit=frequency_unit,
            kind="psd",
            metadata={
                "source_value_unit": source.value_unit,
                "normalization": "density",
                "aggregation": (
                    "none" if block_size is None else "mean_power_density"
                ),
            },
        )
        return self.measurement.add_derived_signal(
            output_name,
            result,
            operation,
            (signal_name,),
            {
                "block_size": block_size,
                "overlap": overlap,
                "window": window,
                "detrend": detrend,
            },
        )

    def signal_statistics(
        self, signal_name, lower=None, upper=None, output_name=None
    ):
        """Return structured signal statistics and optionally store them."""
        source = self._signal(signal_name)
        if source.kind != "signal":
            raise ValueError("signal_statistics requires kind='signal'")
        mask = coordinate_range_mask(source.coordinates, lower, upper)
        if not np.any(mask):
            raise ValueError("coordinate range selects no samples")
        sample_rate = source.inferred_sample_rate()
        spacing = None if sample_rate is None else 1 / sample_rate
        values = calculate_signal_statistics(
            source.values[mask], sample_spacing=spacing
        )
        units = {
            "count": "1",
            "mean": source.value_unit,
            "standard_deviation": source.value_unit,
            "mean_square": _squared_unit(source.value_unit),
            "rms": source.value_unit,
            "absolute_peak": source.value_unit,
            "peak_to_peak": source.value_unit,
            "crest_factor": "1",
            "skewness": "1",
            "kurtosis": "1",
        }
        if "energy" in values:
            units["energy"] = _product_unit(
                _squared_unit(source.value_unit), source.coordinate_unit
            )
        return self._analysis_result(
            "signal_statistics",
            values,
            units,
            signal_name,
            lower,
            upper,
            output_name,
        )

    def amplitude_band_statistics(
        self, signal_name, lower=None, upper=None, output_name=None
    ):
        """Return structured amplitude-band statistics and optionally store them."""
        source = self._signal(signal_name)
        if source.kind != "amplitude_spectrum":
            raise ValueError(
                "amplitude_band_statistics requires kind='amplitude_spectrum'"
            )
        values = amplitude_statistics_in_band(
            source.coordinates, source.values, lower, upper
        )
        units = {
            "count": "1",
            "peak_coordinate": source.coordinate_unit,
            "peak_amplitude": source.value_unit,
            "mean_amplitude": source.value_unit,
            "median_amplitude": source.value_unit,
            "magnitude_integral": _product_unit(
                source.value_unit, source.coordinate_unit
            ),
        }
        return self._analysis_result(
            "amplitude_band_statistics",
            values,
            units,
            signal_name,
            lower,
            upper,
            output_name,
        )

    def psd_band_statistics(
        self, signal_name, lower=None, upper=None, output_name=None
    ):
        """Return structured PSD-band statistics and optionally store them."""
        source = self._signal(signal_name)
        if source.kind != "psd":
            raise ValueError("psd_band_statistics requires kind='psd'")
        values = psd_statistics_in_band(
            source.coordinates, source.values, lower, upper
        )
        source_value_unit = source.metadata.get("source_value_unit")
        units = {
            "count": "1",
            "bin_width": source.coordinate_unit,
            "band_power": _squared_unit(source_value_unit),
            "band_rms": source_value_unit,
            "dominant_coordinate": source.coordinate_unit,
            "power_weighted_center": source.coordinate_unit,
        }
        return self._analysis_result(
            "psd_band_statistics",
            values,
            units,
            signal_name,
            lower,
            upper,
            output_name,
        )

    def _trim_signal(self, source_name, output_name, trim_samples):
        source = self._signal(source_name)
        require_nonnegative_finite_scalar(trim_samples, "trim_samples")
        if isinstance(trim_samples, (bool, np.bool_)) or int(trim_samples) != trim_samples:
            raise ValueError("trim_samples must be a non-negative integer")
        trim_samples = int(trim_samples)
        if 2 * trim_samples >= len(source.values):
            raise ValueError("trim_samples removes the complete signal")
        stop = -trim_samples if trim_samples else None
        values = source.values[trim_samples:stop]
        coordinates = source.coordinates[trim_samples:stop]
        result = Signal(
            values,
            sample_rate=self._sample_rate(source),
            start=coordinates[0],
            value_unit=source.value_unit,
            coordinate_unit=source.coordinate_unit,
            kind=source.kind,
        )
        return self.measurement.add_derived_signal(
            output_name,
            result,
            "trim",
            (source_name,),
            {"trim_samples": trim_samples},
        )

    def envelope_order_spectrum(
        self,
        vibration_name,
        speed_name,
        output_prefix,
        band,
        lowpass,
        block_size,
        trim_samples=0,
        detector="hilbert",
        overlap=0.5,
        window="hann",
        resolution=None,
        oversample=None,
        position_unit="rev",
    ):
        """Run demodulation, explicit trimming, spatial resampling and FFT."""
        envelope_name = f"{output_prefix}_envelope"
        trimmed_envelope_name = f"{output_prefix}_envelope_trimmed"
        trimmed_speed_name = f"{output_prefix}_speed_trimmed"
        spatial_name = f"{output_prefix}_spatial"
        spectrum_name = f"{output_prefix}_order_spectrum"

        self._synchronized(
            self._signal(vibration_name), self._signal(speed_name)
        )
        self.demodulate(
            vibration_name,
            envelope_name,
            band,
            lowpass,
            detector=detector,
        )
        self._trim_signal(envelope_name, trimmed_envelope_name, trim_samples)
        self._trim_signal(speed_name, trimmed_speed_name, trim_samples)
        self.spatial_resample(
            trimmed_envelope_name,
            trimmed_speed_name,
            spatial_name,
            resolution=resolution,
            oversample=oversample,
            position_unit=position_unit,
        )
        return self.amplitude_spectrum(
            spatial_name,
            spectrum_name,
            block_size=block_size,
            overlap=overlap,
            window=window,
        )
