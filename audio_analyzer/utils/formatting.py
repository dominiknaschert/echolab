"""
Formatierungsfunktionen für Anzeige.

Konvertiert numerische Werte in lesbare Strings.
"""

from typing import Optional


def format_time(seconds: float, show_ms: bool = True) -> str:
    """
    Formatiere Zeit in lesbares Format.
    
    Args:
        seconds: Zeit in Sekunden
        show_ms: Zeige Millisekunden
        
    Returns:
        Formatierter String (z.B. "1:23.456" oder "1:23")
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
    Formatiere Frequenz in lesbares Format.
    
    Args:
        hz: Frequenz in Hz
        
    Returns:
        Formatierter String (z.B. "1.5 kHz" oder "250 Hz")
    """
    if hz >= 1000:
        return f"{hz/1000:.1f} kHz"
    else:
        return f"{hz:.0f} Hz"


def format_db(db: float, precision: int = 1) -> str:
    """
    Formatiere dB-Wert.
    
    Args:
        db: Pegel in dB
        precision: Nachkommastellen
        
    Returns:
        Formatierter String (z.B. "-12.3 dB")
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
    Konvertiere Samples zu Zeit-String mit optionaler Sample-Anzeige.
    
    Args:
        samples: Anzahl Samples
        sample_rate: Samplerate
        show_samples: Zeige auch Sample-Anzahl
        
    Returns:
        Formatierter String (z.B. "1:23.456 (65432 samples)")
    """
    time_sec = samples / sample_rate
    time_str = format_time(time_sec)
    
    if show_samples:
        return f"{time_str} ({samples:,} samples)"
    return time_str


def format_sample_rate(sr: int) -> str:
    """
    Formatiere Samplerate.
    
    Args:
        sr: Samplerate in Hz
        
    Returns:
        Formatierter String (z.B. "44.1 kHz" oder "48 kHz")
    """
    if sr % 1000 == 0:
        return f"{sr // 1000} kHz"
    else:
        return f"{sr / 1000:.1f} kHz"


def format_duration(seconds: float) -> str:
    """
    Formatiere Dauer für Anzeige.
    
    Args:
        seconds: Dauer in Sekunden
        
    Returns:
        Formatierter String (z.B. "3:45.2" oder "0:01.5")
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes > 0:
        return f"{minutes}:{secs:05.2f}"
    else:
        return f"0:{secs:05.2f}"


def format_channels(num_channels: int) -> str:
    """
    Formatiere Kanalanzahl.
    
    Args:
        num_channels: Anzahl Kanäle
        
    Returns:
        "Mono" oder "Stereo" oder "X Kanäle"
    """
    if num_channels == 1:
        return "Mono"
    elif num_channels == 2:
        return "Stereo"
    else:
        return f"{num_channels} Kanäle"

