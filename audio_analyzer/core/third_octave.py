"""
IEC 61260 Third-Octave Filterbank

Implements a standards-compliant 1/3-octave filterbank for acoustic analysis.

IMPORTANT: This is a REAL filterbank with IIR filters, not a spectral approximation.

Technical specification:
- Center frequencies according to IEC 61260-1:2014
- Butterworth IIR filters 6th order (Class 1 requirements)
- Bandwidth: fm × (2^(1/6) - 2^(-1/6)) ≈ 0.2316 × fm

Documented limitations:
- IIR filters have nonlinear phase behavior
- Group delay varies with frequency (especially at band edges)
- For phase-critical applications: Zero-phase filtering (filtfilt) optional

Simplifications:
- No calibration to acoustic reference levels
- No A/C weighting integrated
- No real-time optimization
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy import signal
from enum import Enum


class OctaveFraction(Enum):
    """Octave fraction for filterbank."""
    OCTAVE = 1
    THIRD_OCTAVE = 3
    SIXTH_OCTAVE = 6
    TWELFTH_OCTAVE = 12


# IEC 61260-1:2014 Reference center frequencies for 1/3-octaves (in Hz)
# Based on reference frequency 1000 Hz
# fm = 1000 × 10^(n/10) for n = ..., -10, -9, ..., 9, 10, ...
IEC_61260_CENTER_FREQUENCIES = np.array([
    # Low frequencies (often below 20 Hz, not all audible)
    12.5, 16, 20,
    # Bass range
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    # Mid range
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
    # High frequencies
    3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
])


@dataclass
class FilterBandInfo:
    """
    Information about a single filter band.
    
    Documents the technical properties of the filter.
    """
    center_frequency: float  # Hz
    lower_frequency: float   # Hz, -3 dB limit
    upper_frequency: float   # Hz, -3 dB limit
    bandwidth: float         # Hz
    filter_order: int
    filter_type: str        # "butterworth", "bessel", etc.
    
    # Phase and group delay information
    group_delay_at_center: float  # Samples at fc
    phase_linear: bool     # Is the phase linear? (No for IIR)
    
    @property
    def quality_factor(self) -> float:
        """Q-factor of the filter."""
        return self.center_frequency / self.bandwidth


@dataclass
class ThirdOctaveBand:
    """
    Result of filtering a signal through a third-octave band.
    
    Contains the filtered signal and all metadata.
    """
    center_frequency: float
    filtered_signal: np.ndarray
    sample_rate: int
    band_info: FilterBandInfo
    
    def rms(self) -> float:
        """RMS level of filtered signal."""
        return np.sqrt(np.mean(self.filtered_signal ** 2))
    
    def rms_db(self, ref: float = 1.0) -> float:
        """RMS level in dB."""
        rms = self.rms()
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms / ref)
    
    def envelope(self, method: Literal["hilbert", "rms"] = "hilbert") -> np.ndarray:
        """
        Calculate envelope of filtered signal.
        
        Hilbert: Analytic signal, precise envelope
        RMS: Sliding RMS, more robust but less precise
        """
        if method == "hilbert":
            analytic = signal.hilbert(self.filtered_signal)
            return np.abs(analytic)
        else:
            # Sliding RMS with window size proportional to period
            window_size = max(int(self.sample_rate / self.center_frequency * 2), 1)
            kernel = np.ones(window_size) / window_size
            squared = self.filtered_signal ** 2
            return np.sqrt(np.convolve(squared, kernel, mode='same'))


class ThirdOctaveFilterbank:
    """
    IEC 61260 compliant 1/3-octave filterbank.
    
    Creates Butterworth IIR bandpass filters for each third-octave center frequency.
    
    Usage:
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        bands = fb.filter_signal(audio_data)
        
        for band in bands:
            print(f"{band.center_frequency} Hz: {band.rms_db():.1f} dB")
    
    Technical details:
    - Filters are created at initialization for all valid frequencies
    - Frequencies above Nyquist/2 are automatically excluded
    - Filter coefficients are stored as Second-Order Sections (SOS)
      for numerical stability
    """
    
    def __init__(
        self,
        sample_rate: int,
        filter_order: int = 6,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        use_zero_phase: bool = False,
    ):
        """
        Initialize filterbank.
        
        Args:
            sample_rate: Sample rate in Hz
            filter_order: Order of Butterworth filters (default: 6)
            f_min: Minimum center frequency (default: 20 Hz)
            f_max: Maximum center frequency (default: Nyquist/2.5)
            use_zero_phase: Zero-phase filtering (filtfilt) for linear phase
        """
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.use_zero_phase = use_zero_phase
        self.nyquist = sample_rate / 2
        
        # Determine valid frequency limits
        if f_max is None:
            # Upper band limit must be below Nyquist
            # For 1/3-octave, upper limit is fm × 2^(1/6) ≈ 1.122 × fm
            f_max = self.nyquist / 1.2
        
        # Filter valid center frequencies
        self.center_frequencies = IEC_61260_CENTER_FREQUENCIES[
            (IEC_61260_CENTER_FREQUENCIES >= f_min) &
            (IEC_61260_CENTER_FREQUENCIES <= f_max)
        ].copy()
        
        # Create filters for each center frequency
        self._filters: dict[float, tuple[np.ndarray, FilterBandInfo]] = {}
        self._create_filters()
    
    def _create_filters(self) -> None:
        """Create all bandpass filters."""
        for fc in self.center_frequencies:
            sos, band_info = self._design_band_filter(fc)
            self._filters[fc] = (sos, band_info)
    
    def _design_band_filter(
        self,
        center_freq: float,
    ) -> tuple[np.ndarray, FilterBandInfo]:
        """
        Design Butterworth bandpass filter for a center frequency.
        
        Band limits according to IEC 61260:
        - Lower limit: fc / 2^(1/6)
        - Upper limit: fc × 2^(1/6)
        
        Args:
            center_freq: Center frequency in Hz
            
        Returns:
            Tuple of (SOS coefficients, band information)
        """
        # IEC 61260 band limits for 1/3-octave
        factor = 2 ** (1/6)  # ≈ 1.1225
        f_low = center_freq / factor
        f_high = center_freq * factor
        
        # Normalize to Nyquist frequency for scipy
        low_normalized = f_low / self.nyquist
        high_normalized = f_high / self.nyquist
        
        # Safety check
        if high_normalized >= 1.0:
            high_normalized = 0.99
        if low_normalized <= 0:
            low_normalized = 0.001
        
        # Design Butterworth filter as SOS for stability
        sos = signal.butter(
            self.filter_order,
            [low_normalized, high_normalized],
            btype='bandpass',
            output='sos',
        )
        
        # Calculate group delay at center frequency
        # Suppress warning: For some filters, the calculation can be numerically unstable,
        # but this has no effect on the filtering itself
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*denominator.*extremely small.*')
            try:
                b, a = signal.sos2tf(sos)
                w, gd = signal.group_delay(
                    (b, a),
                    w=[2 * np.pi * center_freq / self.sample_rate],
                )
                group_delay_samples = float(gd[0]) if len(gd) > 0 else 0.0
            except (ValueError, np.linalg.LinAlgError):
                # Fallback: Estimate group delay based on filter order
                # For Butterworth bandpass: approx. filter_order / (2 * bandwidth)
                bandwidth_hz = f_high - f_low
                group_delay_samples = self.filter_order / (2 * bandwidth_hz * (1 / self.sample_rate))
        
        band_info = FilterBandInfo(
            center_frequency=center_freq,
            lower_frequency=f_low,
            upper_frequency=f_high,
            bandwidth=f_high - f_low,
            filter_order=self.filter_order,
            filter_type="butterworth",
            group_delay_at_center=float(group_delay_samples),
            phase_linear=self.use_zero_phase,
        )
        
        return sos, band_info
    
    def filter_signal(
        self,
        data: np.ndarray,
        frequencies: Optional[list[float]] = None,
    ) -> list[ThirdOctaveBand]:
        """
        Filter signal through all (or selected) third-octave bands.
        
        Args:
            data: Audio signal (1D)
            frequencies: Optional, list of specific center frequencies
            
        Returns:
            List of ThirdOctaveBand objects
        """
        if data.ndim != 1:
            raise ValueError("Signal must be 1D (Mono)")
        
        if frequencies is None:
            frequencies = list(self.center_frequencies)
        
        results = []
        for fc in frequencies:
            if fc not in self._filters:
                raise ValueError(f"Frequency {fc} Hz not available in filterbank")
            
            sos, band_info = self._filters[fc]
            
            if self.use_zero_phase:
                # Zero-phase filtering: double order, linear phase
                filtered = signal.sosfiltfilt(sos, data)
            else:
                # Standard IIR filtering: causal filtering
                filtered = signal.sosfilt(sos, data)
            
            results.append(ThirdOctaveBand(
                center_frequency=fc,
                filtered_signal=filtered,
                sample_rate=self.sample_rate,
                band_info=band_info,
            ))
        
        return results
    
    def filter_single_band(
        self,
        data: np.ndarray,
        center_frequency: float,
    ) -> ThirdOctaveBand:
        """Filter signal through single third-octave band."""
        results = self.filter_signal(data, [center_frequency])
        return results[0]
    
    def compute_time_varying_levels(
        self,
        data: np.ndarray,
        time_resolution_ms: float = 10.0,
        level_type: Literal["rms", "peak", "envelope"] = "rms",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate time-varying levels for all third-octave bands.
        
        This is the core function for impulse response analysis.
        
        Args:
            data: Audio signal (1D)
            time_resolution_ms: Time resolution in milliseconds
            level_type: Type of level calculation
            
        Returns:
            Tuple of:
            - levels: Shape (num_bands, num_time_frames), level per band and time
            - times: Time axis in seconds
            - frequencies: Center frequencies
        """
        # Calculate frame parameters
        samples_per_frame = int(self.sample_rate * time_resolution_ms / 1000)
        num_frames = len(data) // samples_per_frame
        
        # Filter all bands
        bands = self.filter_signal(data)
        
        # Initialize result array
        levels = np.zeros((len(bands), num_frames))
        
        for i, band in enumerate(bands):
            for frame in range(num_frames):
                start = frame * samples_per_frame
                end = start + samples_per_frame
                segment = band.filtered_signal[start:end]
                
                if level_type == "rms":
                    levels[i, frame] = np.sqrt(np.mean(segment ** 2))
                elif level_type == "peak":
                    levels[i, frame] = np.max(np.abs(segment))
                elif level_type == "envelope":
                    # Peak value of envelope
                    env = np.abs(signal.hilbert(segment))
                    levels[i, frame] = np.max(env)
        
        # Time axis
        times = np.arange(num_frames) * time_resolution_ms / 1000
        
        # Frequencies
        frequencies = np.array([b.center_frequency for b in bands])
        
        return levels, times, frequencies
    
    def get_band_info(self, center_frequency: float) -> FilterBandInfo:
        """Get technical information about a band."""
        if center_frequency not in self._filters:
            raise ValueError(f"Frequency {center_frequency} Hz not available")
        return self._filters[center_frequency][1]
    
    def get_all_band_info(self) -> list[FilterBandInfo]:
        """Get technical information about all bands."""
        return [self._filters[fc][1] for fc in self.center_frequencies]
    
    @property
    def num_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.center_frequencies)
    
    def frequency_response(
        self,
        center_frequency: float,
        num_points: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate frequency response of a filter.
        
        For documentation and visualization of filter behavior.
        
        Returns:
            Tuple of (frequencies in Hz, magnitude in dB, phase in degrees)
        """
        if center_frequency not in self._filters:
            raise ValueError(f"Frequency {center_frequency} Hz not available")
        
        sos, _ = self._filters[center_frequency]
        
        # Calculate frequency response
        w, h = signal.sosfreqz(sos, worN=num_points, fs=self.sample_rate)
        
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        phase_deg = np.angle(h, deg=True)
        
        return w, magnitude_db, phase_deg


def compute_third_octave_spectrum(
    data: np.ndarray,
    sample_rate: int,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate 1/3-octave spectrum (time-averaged) of a signal.
    
    Shortcut function for simple spectral analysis.
    
    Args:
        data: Audio signal (1D)
        sample_rate: Sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency
        
    Returns:
        Tuple of (center frequencies, RMS level in dB)
    """
    fb = ThirdOctaveFilterbank(sample_rate, f_min=f_min, f_max=f_max)
    bands = fb.filter_signal(data)
    
    frequencies = np.array([b.center_frequency for b in bands])
    levels_db = np.array([b.rms_db() for b in bands])
    
    return frequencies, levels_db

