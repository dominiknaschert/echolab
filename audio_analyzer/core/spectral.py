"""
Spectral Analysis Module

Computes spectrograms with configurable parameters.
Optimized for analysis, not for aesthetic display.

Technical assumptions:
- STFT-based spectrogram computation
- Logarithmic frequency axis for display
- All parameters are explicitly configurable
- No smoothing or interpolation without user decision
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy import signal


@dataclass
class SpectrogramConfig:
    """
    Configuration for spectrogram computation.
    
    All parameters must be explicitly specified.
    There are no "automatic" settings.
    
    Attributes:
        fft_size: FFT size (should be power of 2)
        hop_size: Step size in samples (determines time resolution)
        window: Window function
        overlap_percent: Alternative to hop_size, calculates hop automatically
    """
    fft_size: int = 2048
    hop_size: Optional[int] = None
    window: Literal["hann", "hamming", "blackman", "kaiser", "rectangular"] = "hann"
    overlap_percent: float = 75.0
    kaiser_beta: float = 14.0
    
    def __post_init__(self):
        """Calculate hop_size from overlap if not specified."""
        if self.hop_size is None:
            self.hop_size = int(self.fft_size * (1 - self.overlap_percent / 100))
        
        # Validation
        if self.fft_size < 32:
            raise ValueError("FFT size must be at least 32")
        if self.hop_size < 1:
            raise ValueError("Hop size must be at least 1")
        if self.hop_size > self.fft_size:
            raise ValueError("Hop size must not be greater than FFT size")
    
    @property
    def effective_overlap(self) -> float:
        """Actual overlap in percent."""
        return (1 - self.hop_size / self.fft_size) * 100
    
    def frequency_resolution(self, sample_rate: int) -> float:
        """Frequency resolution in Hz."""
        return sample_rate / self.fft_size
    
    def time_resolution(self, sample_rate: int) -> float:
        """Time resolution in seconds."""
        return self.hop_size / sample_rate


@dataclass 
class SpectrogramResult:
    """
    Result of a spectrogram computation.
    
    Contains all necessary information for display and analysis.
    
    Attributes:
        magnitude: Magnitude values, Shape: (frequencies, time_frames)
        phase: Phase values in radians, Shape: (frequencies, time_frames)
        frequencies: Frequency axis in Hz
        times: Time axis in seconds
        config: Used configuration
        sample_rate: Sample rate of input data
    """
    magnitude: np.ndarray
    phase: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    config: SpectrogramConfig
    sample_rate: int
    
    def magnitude_db(self, ref: float = 1.0, min_db: float = -120.0) -> np.ndarray:
        """
        Magnitude in dB.
        
        Args:
            ref: Reference value (1.0 for dBFS)
            min_db: Minimum dB value (to avoid log(0))
            
        Returns:
            Magnitude in dB, Shape: (frequencies, time_frames)
        """
        # Avoid log(0)
        mag = np.maximum(self.magnitude, 10 ** (min_db / 20) * ref)
        return 20 * np.log10(mag / ref)
    
    def power_db(self, ref: float = 1.0, min_db: float = -120.0) -> np.ndarray:
        """Power spectrum in dB."""
        power = self.magnitude ** 2
        power = np.maximum(power, 10 ** (min_db / 10) * ref)
        return 10 * np.log10(power / ref)


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    config: Optional[SpectrogramConfig] = None,
) -> SpectrogramResult:
    """
    Compute spectrogram of an audio signal.
    
    Uses Short-Time Fourier Transform (STFT).
    
    Technical details:
    - Windowing reduces spectral leakage
    - Overlap improves time resolution
    - Nyquist frequency limits frequency axis to sample_rate/2
    
    Frequency-time resolution trade-off (Heisenberg):
    - Large FFT: Good frequency resolution, poor time resolution
    - Small FFT: Good time resolution, poor frequency resolution
    
    Args:
        data: Audio signal (1D mono or single channel)
        sample_rate: Sample rate in Hz
        config: Spectrogram configuration
        
    Returns:
        SpectrogramResult with all data
    """
    if config is None:
        config = SpectrogramConfig()
    
    # Ensure we have 1D data
    if data.ndim != 1:
        raise ValueError("Spectrogram requires 1D signal (mono or single channel)")
    
    # Create window
    window = _create_window(config.window, config.fft_size, config.kaiser_beta)
    
    # Compute STFT with scipy
    frequencies, times, stft_result = signal.stft(
        data,
        fs=sample_rate,
        window=window,
        nperseg=config.fft_size,
        noverlap=config.fft_size - config.hop_size,
        nfft=config.fft_size,
        return_onesided=True,
        boundary='zeros',
        padded=True,
    )
    
    # Extract magnitude and phase
    magnitude = np.abs(stft_result)
    phase = np.angle(stft_result)
    
    return SpectrogramResult(
        magnitude=magnitude,
        phase=phase,
        frequencies=frequencies,
        times=times,
        config=config,
        sample_rate=sample_rate,
    )


def compute_power_spectrum(
    data: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
    window: Literal["hann", "hamming", "blackman", "kaiser", "rectangular"] = "hann",
    average_method: Literal["mean", "median"] = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute averaged power spectrum (Welch method).
    
    For overall spectrum of a signal, not time-resolved.
    
    Args:
        data: Audio signal (1D)
        sample_rate: Sample rate
        fft_size: FFT size
        window: Window function
        average_method: Averaging method
        
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    if data.ndim != 1:
        raise ValueError("Power spectrum requires 1D signal")
    
    frequencies, psd = signal.welch(
        data,
        fs=sample_rate,
        window=window,
        nperseg=fft_size,
        noverlap=fft_size // 2,
        nfft=fft_size,
        average=average_method,
    )
    
    return frequencies, psd


def frequency_to_log_scale(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    num_bins: int = 512,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert linear spectrogram to logarithmic frequency axis.
    
    For better visual display, especially in the bass range.
    
    Technical details:
    - Interpolation between adjacent bins
    - Frequencies below f_min are truncated
    - Logarithmically equidistant frequency bins
    
    Args:
        magnitude: Magnitude values, Shape: (frequencies, time)
        frequencies: Linear frequency axis in Hz
        num_bins: Number of logarithmic bins
        f_min: Minimum frequency (typically 20 Hz)
        f_max: Maximum frequency (default: Nyquist)
        
    Returns:
        Tuple of (logarithmic magnitudes, logarithmic frequencies)
    """
    if f_max is None:
        f_max = frequencies[-1]
    
    # Logarithmic frequency axis
    log_frequencies = np.logspace(np.log10(f_min), np.log10(f_max), num_bins)
    
    # Interpolation for each time frame
    log_magnitude = np.zeros((num_bins, magnitude.shape[1]))
    
    for t in range(magnitude.shape[1]):
        log_magnitude[:, t] = np.interp(
            log_frequencies,
            frequencies,
            magnitude[:, t],
            left=0,
            right=0,
        )
    
    return log_magnitude, log_frequencies


def _create_window(
    window_type: str,
    size: int,
    kaiser_beta: float = 14.0,
) -> np.ndarray:
    """Create window function."""
    if window_type == "hann":
        return signal.windows.hann(size)
    elif window_type == "hamming":
        return signal.windows.hamming(size)
    elif window_type == "blackman":
        return signal.windows.blackman(size)
    elif window_type == "kaiser":
        return signal.windows.kaiser(size, kaiser_beta)
    elif window_type == "rectangular":
        return np.ones(size)
    else:
        raise ValueError(f"Unbekannte Fensterfunktion: {window_type}")

