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



