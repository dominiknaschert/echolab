"""
Allgemeine Signalverarbeitung

Enthält grundlegende Funktionen zur Signalmanipulation.
Alle Funktionen sind EXPLIZIT - keine automatischen Konvertierungen.

Technische Annahmen:
- Resampling verwendet scipy.signal.resample_poly für Anti-Aliasing
- Downmix erfolgt als arithmetisches Mittel (keine Energiekompensation)
- Alle Operationen arbeiten auf Kopien, Originaldaten bleiben unverändert
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
    Resample Audiodaten auf neue Samplerate.
    
    Verwendet scipy.signal.resample_poly mit automatischer Anti-Aliasing-Filterung.
    
    Technische Details:
    - Polyphasen-Resampling für effiziente Berechnung
    - Anti-Aliasing-Filter: Kaiser-Window FIR
    - Gruppenlaufzeit wird durch Padding kompensiert
    
    Args:
        data: Audiodaten (1D oder 2D)
        original_sr: Original-Samplerate
        target_sr: Ziel-Samplerate
        
    Returns:
        Resampelte Audiodaten (gleiche Dimensionalität)
    """
    if original_sr == target_sr:
        return data.copy()
    
    # Bestimme Upsampling/Downsampling-Faktoren
    gcd = np.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    
    if data.ndim == 1:
        return signal.resample_poly(data, up, down).astype(data.dtype)
    else:
        # Resample jeden Kanal separat
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
    Konvertiere Stereo zu Mono.
    
    KEIN automatischer Downmix - diese Funktion muss explizit aufgerufen werden.
    
    Methoden:
    - average: (L + R) / 2 - Standard, keine Energiekompensation
    - left: Nur linker Kanal
    - right: Nur rechter Kanal
    - mid: (L + R) / 2 - Identisch zu average, semantisch "Mitte"
    - side: (L - R) / 2 - Seitensignal (Stereodifferenz)
    
    Hinweis zu Energiekompensation:
    Bei korreliertem Material (z.B. zentrierte Instrumente) würde der
    average-Downmix zu 6 dB Pegelreduktion führen. Diese Korrektur wird
    NICHT automatisch angewendet, da sie von der Korrelation abhängt.
    
    Args:
        data: Stereo-Audiodaten, Shape: (samples, 2)
        method: Downmix-Methode
        
    Returns:
        Mono-Audiodaten, Shape: (samples,)
    """
    if data.ndim == 1:
        return data.copy()  # Bereits Mono
    
    if data.shape[1] != 2:
        raise ValueError(f"Erwartet Stereo (2 Kanäle), erhalten: {data.shape[1]}")
    
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
        raise ValueError(f"Unbekannte Methode: {method}")


def normalize_audio(
    data: np.ndarray,
    target_peak: float = 1.0,
    reference: Literal["peak", "rms"] = "peak",
    target_rms_db: float = -20.0,
) -> tuple[np.ndarray, float]:
    """
    Normalisiere Audiodaten.
    
    Diese Funktion verändert die Amplitude des Signals.
    Der Normalisierungsfaktor wird zurückgegeben für Dokumentation.
    
    Args:
        data: Audiodaten
        target_peak: Ziel-Peakwert bei peak-Normalisierung (0.0-1.0)
        reference: Normalisierungsreferenz ("peak" oder "rms")
        target_rms_db: Ziel-RMS in dB bei RMS-Normalisierung
        
    Returns:
        Tuple aus (normalisierte Daten, angewendeter Faktor)
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
        raise ValueError(f"Unbekannte Referenz: {reference}")
    
    return data * factor, factor


def compute_rms(data: np.ndarray, as_db: bool = False) -> float:
    """
    Berechne RMS (Root Mean Square) des Signals.
    
    Bei Mehrkanal-Audio wird der RMS über alle Kanäle berechnet.
    
    Args:
        data: Audiodaten
        as_db: Wenn True, Rückgabe in dB (Referenz: 1.0)
        
    Returns:
        RMS-Wert (linear oder dB)
    """
    rms = np.sqrt(np.mean(data ** 2))
    
    if as_db:
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)
    
    return rms


def compute_peak(data: np.ndarray, as_db: bool = False) -> float:
    """
    Berechne Spitzenwert (absoluter Maximalwert) des Signals.
    
    Args:
        data: Audiodaten
        as_db: Wenn True, Rückgabe in dB (Referenz: 1.0)
        
    Returns:
        Peak-Wert (linear oder dB)
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
    Wende Fensterfunktion auf Signal an.
    
    Fensterfunktionen reduzieren spektrale Leckage bei der FFT.
    
    Eigenschaften der Fenster:
    - hann: Guter Kompromiss, -31.5 dB Seitenkeulen
    - hamming: Bessere Seitenkeulendämpfung (-43 dB), breitere Hauptkeule
    - blackman: Sehr gute Dämpfung (-58 dB), breiteste Hauptkeule
    - kaiser: Einstellbar via beta, höher = mehr Dämpfung
    - rectangular: Kein Fenster, maximale Leckage
    
    Args:
        data: Eingangssignal
        window_type: Art der Fensterfunktion
        kaiser_beta: Beta-Parameter für Kaiser-Fenster
        
    Returns:
        Gefenstertes Signal
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
    Berechne Crest Factor (Verhältnis Peak zu RMS).
    
    Der Crest Factor beschreibt die "Spitzigkeit" eines Signals.
    - Sinuston: ~1.414 (3 dB)
    - Weißes Rauschen: ~3-4
    - Stark komprimierte Musik: ~2-3
    - Dynamische Klassik: ~10-20
    
    Returns:
        Crest Factor (linear, nicht dB)
    """
    peak = compute_peak(data)
    rms = compute_rms(data)
    
    if rms == 0:
        return np.inf
    
    return peak / rms


def compute_dc_offset(data: np.ndarray) -> float:
    """
    Berechne DC-Offset (Gleichanteil) des Signals.
    
    Ein signifikanter DC-Offset (>0.01) kann auf Aufnahmeprobleme
    oder beschädigte Dateien hinweisen.
    
    Returns:
        DC-Offset als Mittelwert der Samples
    """
    return np.mean(data)


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Entferne DC-Offset aus Signal.
    
    Subtrahiert den Mittelwert von allen Samples.
    Dies ist eine explizite Operation, keine automatische Korrektur.
    
    Returns:
        Signal ohne DC-Offset
    """
    return data - np.mean(data, axis=0)

