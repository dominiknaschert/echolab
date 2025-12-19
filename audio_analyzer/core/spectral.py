"""
Spektralanalyse-Modul

Berechnet Spektrogramme mit konfigurierbaren Parametern.
Optimiert für Analyse, nicht für ästhetische Darstellung.

Technische Annahmen:
- STFT-basierte Spektrogrammberechnung
- Logarithmische Frequenzachse für Darstellung
- Alle Parameter sind explizit konfigurierbar
- Keine Glättung oder Interpolation ohne Nutzerentscheidung
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy import signal


@dataclass
class SpectrogramConfig:
    """
    Konfiguration für Spektrogramm-Berechnung.
    
    Alle Parameter sind explizit anzugeben.
    Es gibt keine "automatischen" Einstellungen.
    
    Attributes:
        fft_size: FFT-Größe (sollte Potenz von 2 sein)
        hop_size: Schrittweite in Samples (bestimmt Zeitauflösung)
        window: Fensterfunktion
        overlap_percent: Alternative zu hop_size, berechnet hop automatisch
    """
    fft_size: int = 2048
    hop_size: Optional[int] = None
    window: Literal["hann", "hamming", "blackman", "kaiser", "rectangular"] = "hann"
    overlap_percent: float = 75.0
    kaiser_beta: float = 14.0
    
    def __post_init__(self):
        """Berechne hop_size aus overlap wenn nicht angegeben."""
        if self.hop_size is None:
            self.hop_size = int(self.fft_size * (1 - self.overlap_percent / 100))
        
        # Validierung
        if self.fft_size < 32:
            raise ValueError("FFT-Größe muss mindestens 32 sein")
        if self.hop_size < 1:
            raise ValueError("Hop-Size muss mindestens 1 sein")
        if self.hop_size > self.fft_size:
            raise ValueError("Hop-Size darf nicht größer als FFT-Größe sein")
    
    @property
    def effective_overlap(self) -> float:
        """Tatsächliche Überlappung in Prozent."""
        return (1 - self.hop_size / self.fft_size) * 100
    
    def frequency_resolution(self, sample_rate: int) -> float:
        """Frequenzauflösung in Hz."""
        return sample_rate / self.fft_size
    
    def time_resolution(self, sample_rate: int) -> float:
        """Zeitauflösung in Sekunden."""
        return self.hop_size / sample_rate


@dataclass 
class SpectrogramResult:
    """
    Ergebnis einer Spektrogramm-Berechnung.
    
    Enthält alle notwendigen Informationen für Darstellung und Analyse.
    
    Attributes:
        magnitude: Magnitudenwerte, Shape: (frequencies, time_frames)
        phase: Phasenwerte in Radiant, Shape: (frequencies, time_frames)
        frequencies: Frequenzachse in Hz
        times: Zeitachse in Sekunden
        config: Verwendete Konfiguration
        sample_rate: Samplerate der Eingangsdaten
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
            ref: Referenzwert (1.0 für dBFS)
            min_db: Minimaler dB-Wert (für log(0)-Vermeidung)
            
        Returns:
            Magnitude in dB, Shape: (frequencies, time_frames)
        """
        # Vermeide log(0)
        mag = np.maximum(self.magnitude, 10 ** (min_db / 20) * ref)
        return 20 * np.log10(mag / ref)
    
    def power_db(self, ref: float = 1.0, min_db: float = -120.0) -> np.ndarray:
        """Leistungsspektrum in dB."""
        power = self.magnitude ** 2
        power = np.maximum(power, 10 ** (min_db / 10) * ref)
        return 10 * np.log10(power / ref)


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    config: Optional[SpectrogramConfig] = None,
) -> SpectrogramResult:
    """
    Berechne Spektrogramm eines Audiosignals.
    
    Verwendet Short-Time Fourier Transform (STFT).
    
    Technische Details:
    - Fensterung reduziert spektrale Leckage
    - Überlappung verbessert Zeitauflösung
    - Nyquist-Frequenz begrenzt Frequenzachse auf sample_rate/2
    
    Frequenz-Zeit-Auflösungs-Kompromiss (Heisenberg):
    - Große FFT: Gute Frequenzauflösung, schlechte Zeitauflösung
    - Kleine FFT: Gute Zeitauflösung, schlechte Frequenzauflösung
    
    Args:
        data: Audiosignal (1D Mono oder einzelner Kanal)
        sample_rate: Samplerate in Hz
        config: Spektrogramm-Konfiguration
        
    Returns:
        SpectrogramResult mit allen Daten
    """
    if config is None:
        config = SpectrogramConfig()
    
    # Stelle sicher, dass wir 1D-Daten haben
    if data.ndim != 1:
        raise ValueError("Spektrogramm benötigt 1D-Signal (Mono oder einzelner Kanal)")
    
    # Erstelle Fenster
    window = _create_window(config.window, config.fft_size, config.kaiser_beta)
    
    # Berechne STFT mit scipy
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
    
    # Extrahiere Magnitude und Phase
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
    Berechne gemitteltes Leistungsspektrum (Welch-Methode).
    
    Für Gesamtspektrum eines Signals, nicht zeitaufgelöst.
    
    Args:
        data: Audiosignal (1D)
        sample_rate: Samplerate
        fft_size: FFT-Größe
        window: Fensterfunktion
        average_method: Mittelungsmethode
        
    Returns:
        Tuple aus (Frequenzen, Leistungsdichte)
    """
    if data.ndim != 1:
        raise ValueError("Leistungsspektrum benötigt 1D-Signal")
    
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
    Konvertiere lineares Spektrogramm zu logarithmischer Frequenzachse.
    
    Für bessere visuelle Darstellung, besonders im Bassbereich.
    
    Technische Details:
    - Interpolation zwischen benachbarten Bins
    - Frequenzen unter f_min werden abgeschnitten
    - Logarithmisch äquidistante Frequenzbins
    
    Args:
        magnitude: Magnitudenwerte, Shape: (frequencies, time)
        frequencies: Lineare Frequenzachse in Hz
        num_bins: Anzahl der logarithmischen Bins
        f_min: Minimale Frequenz (typisch 20 Hz)
        f_max: Maximale Frequenz (default: Nyquist)
        
    Returns:
        Tuple aus (logarithmische Magnitudes, logarithmische Frequenzen)
    """
    if f_max is None:
        f_max = frequencies[-1]
    
    # Logarithmische Frequenzachse
    log_frequencies = np.logspace(np.log10(f_min), np.log10(f_max), num_bins)
    
    # Interpolation für jede Zeitframe
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
    """Erstelle Fensterfunktion."""
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

