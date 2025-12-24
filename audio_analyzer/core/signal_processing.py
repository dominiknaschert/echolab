"""
General Signal Processing

Contains fundamental functions for signal manipulation.
All functions are EXPLICIT - no automatic conversions.

Technical assumptions:
- Resampling uses scipy.signal.resample_poly for anti-aliasing
- Downmix is performed as arithmetic mean (no energy compensation)
- All operations work on copies, original data remains unchanged
"""

import numpy as np
from scipy import signal
from typing import Literal, Optional


def resample_audio(
    data: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio data to new sample rate.
    
    Uses scipy.signal.resample_poly with automatic anti-aliasing filtering.
    
    Technical details:
    - Polyphase resampling for efficient computation
    - Anti-aliasing filter: Kaiser window FIR
    - Group delay is compensated by padding
    
    Args:
        data: Audio data (1D or 2D)
        original_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data (same dimensionality)
    """
    if original_sr == target_sr:
        return data.copy()
    
    # Determine upsampling/downsampling factors
    gcd = np.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    
    if data.ndim == 1:
        return signal.resample_poly(data, up, down).astype(data.dtype)
    else:
        # Resample each channel separately
        result = np.zeros(
            (int(len(data) * target_sr / original_sr), data.shape[1]),
            dtype=data.dtype
        )
        for ch in range(data.shape[1]):
            result[:, ch] = signal.resample_poly(data[:, ch], up, down)
        return result


def downmix_to_mono(
    data: np.ndarray,
    method: Literal["average", "left", "right", "side", "mid"] = "average"
) -> np.ndarray:
    """
    Convert stereo to mono.
    
    NO automatic downmix - this function must be called explicitly.
    
    Methods:
    - average: (L + R) / 2 - Standard, no energy compensation
    - left: Left channel only
    - right: Right channel only
    - mid: (L + R) / 2 - Identical to average, semantically "mid"
    - side: (L - R) / 2 - Side signal (stereo difference)
    
    Note on energy compensation:
    For correlated material (e.g., centered instruments), the
    average downmix would lead to 6 dB level reduction. This correction is
    NOT automatically applied, as it depends on correlation.
    
    Args:
        data: Stereo audio data, Shape: (samples, 2)
        method: Downmix method
        
    Returns:
        Mono audio data, Shape: (samples,)
    """
    if data.ndim == 1:
        return data.copy()  # Already mono
    
    if data.shape[1] != 2:
        raise ValueError(f"Expected stereo (2 channels), got: {data.shape[1]}")
    
    left = data[:, 0]
    right = data[:, 1]
    
    if method == "average" or method == "mid":
        return (left + right) / 2
    elif method == "left":
        return left.copy()
    elif method == "right":
        return right.copy()
    elif method == "side":
        return (left - right) / 2
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_audio(
    data: np.ndarray,
    target_peak: float = 1.0,
    reference: Literal["peak", "rms"] = "peak",
    target_rms_db: float = -20.0,
) -> tuple[np.ndarray, float]:
    """
    Normalize audio data.
    
    This function modifies the amplitude of the signal.
    The normalization factor is returned for documentation.
    
    Args:
        data: Audio data
        target_peak: Target peak value for peak normalization (0.0-1.0)
        reference: Normalization reference ("peak" or "rms")
        target_rms_db: Target RMS in dB for RMS normalization
        
    Returns:
        Tuple of (normalized data, applied factor)
    """
    if reference == "peak":
        current_peak = compute_peak(data)
        if current_peak == 0:
            return data.copy(), 1.0
        factor = target_peak / current_peak
    elif reference == "rms":
        current_rms_db = compute_rms(data, as_db=True)
        factor_db = target_rms_db - current_rms_db
        factor = 10 ** (factor_db / 20)
    else:
        raise ValueError(f"Unknown reference: {reference}")
    
    return data * factor, factor


def compute_rms(data: np.ndarray, as_db: bool = False) -> float:
    """
    Compute RMS (Root Mean Square) of the signal.
    
    For multi-channel audio, RMS is computed over all channels.
    
    Args:
        data: Audio data
        as_db: If True, return in dB (reference: 1.0)
        
    Returns:
        RMS value (linear or dB)
    """
    rms = np.sqrt(np.mean(data ** 2))
    
    if as_db:
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)
    
    return rms


def compute_peak(data: np.ndarray, as_db: bool = False) -> float:
    """
    Compute peak value (absolute maximum) of the signal.
    
    Args:
        data: Audio data
        as_db: If True, return in dB (reference: 1.0)
        
    Returns:
        Peak value (linear or dB)
    """
    peak = np.max(np.abs(data))
    
    if as_db:
        if peak == 0:
            return -np.inf
        return 20 * np.log10(peak)
    
    return peak


def apply_window(
    data: np.ndarray,
    window_type: Literal["hann", "hamming", "blackman", "kaiser", "rectangular"] = "hann",
    kaiser_beta: float = 14.0,
) -> np.ndarray:
    """
    Apply window function to signal.
    
    Window functions reduce spectral leakage in FFT.
    
    Window properties:
    - hann: Good compromise, -31.5 dB side lobes
    - hamming: Better side lobe suppression (-43 dB), wider main lobe
    - blackman: Very good suppression (-58 dB), widest main lobe
    - kaiser: Adjustable via beta, higher = more suppression
    - rectangular: No window, maximum leakage
    
    Args:
        data: Input signal
        window_type: Type of window function
        kaiser_beta: Beta parameter for Kaiser window
        
    Returns:
        Windowed signal
    """
    n = len(data) if data.ndim == 1 else data.shape[0]
    
    if window_type == "hann":
        window = signal.windows.hann(n)
    elif window_type == "hamming":
        window = signal.windows.hamming(n)
    elif window_type == "blackman":
        window = signal.windows.blackman(n)
    elif window_type == "kaiser":
        window = signal.windows.kaiser(n, kaiser_beta)
    elif window_type == "rectangular":
        window = np.ones(n)
    else:
        raise ValueError(f"Unbekannte Fensterfunktion: {window_type}")
    
    if data.ndim == 1:
        return data * window
    else:
        return data * window[:, np.newaxis]


def compute_crest_factor(data: np.ndarray) -> float:
    """
    Compute crest factor (ratio of peak to RMS).
    
    The crest factor describes the "peakiness" of a signal.
    - Sine tone: ~1.414 (3 dB)
    - White noise: ~3-4
    - Heavily compressed music: ~2-3
    - Dynamic classical: ~10-20
    
    Returns:
        Crest factor (linear, not dB)
    """
    peak = compute_peak(data)
    rms = compute_rms(data)
    
    if rms == 0:
        return np.inf
    
    return peak / rms


def compute_dc_offset(data: np.ndarray) -> float:
    """
    Compute DC offset (DC component) of the signal.
    
    A significant DC offset (>0.01) may indicate recording problems
    or corrupted files.
    
    Returns:
        DC offset as mean of samples
    """
    return np.mean(data)


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from signal.
    
    Subtracts the mean from all samples.
    This is an explicit operation, not an automatic correction.
    
    Returns:
        Signal without DC offset
    """
    return data - np.mean(data, axis=0)

