"""
Formatting functions for display.

Converts numerical values to readable strings.
"""

from typing import Optional


def format_time(seconds: float, show_ms: bool = True) -> str:
    """
    Format time in readable format.
    
    Args:
        seconds: Time in seconds
        show_ms: Show milliseconds
        
    Returns:
        Formatted string (e.g. "1:23.456" or "1:23")
    """
    if seconds < 0:
        sign = "-"
        seconds = abs(seconds)
    else:
        sign = ""
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if show_ms:
        return f"{sign}{minutes}:{secs:06.3f}"
    else:
        return f"{sign}{minutes}:{int(secs):02d}"


def format_frequency(hz: float) -> str:
    """
    Format frequency in readable format.
    
    Args:
        hz: Frequency in Hz
        
    Returns:
        Formatted string (e.g. "1.5 kHz" or "250 Hz")
    """
    if hz >= 1000:
        return f"{hz/1000:.1f} kHz"
    else:
        return f"{hz:.0f} Hz"


def format_db(db: float, precision: int = 1) -> str:
    """
    Format dB value.
    
    Args:
        db: Level in dB
        precision: Decimal places
        
    Returns:
        Formatted string (e.g. "-12.3 dB")
    """
    if db == float('-inf'):
        return "-∞ dB"
    return f"{db:.{precision}f} dB"


def samples_to_time_str(
    samples: int,
    sample_rate: int,
    show_samples: bool = True,
) -> str:
    """
    Convert samples to time string with optional sample display.
    
    Args:
        samples: Number of samples
        sample_rate: Sample rate
        show_samples: Also show sample count
        
    Returns:
        Formatted string (e.g. "1:23.456 (65432 samples)")
    """
    time_sec = samples / sample_rate
    time_str = format_time(time_sec)
    
    if show_samples:
        return f"{time_str} ({samples:,} samples)"
    return time_str


def format_sample_rate(sr: int) -> str:
    """
    Format sample rate.
    
    Args:
        sr: Sample rate in Hz
        
    Returns:
        Formatted string (e.g. "44.1 kHz" or "48 kHz")
    """
    if sr % 1000 == 0:
        return f"{sr // 1000} kHz"
    else:
        return f"{sr / 1000:.1f} kHz"


def format_channels(num_channels: int) -> str:
    """
    Format channel count.
    
    Args:
        num_channels: Number of channels
        
    Returns:
        "Mono" or "Stereo" or "X Channels"
    """
    if num_channels == 1:
        return "Mono"
    elif num_channels == 2:
        return "Stereo"
    else:
        return f"{num_channels} Channels"

