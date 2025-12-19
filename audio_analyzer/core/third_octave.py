"""
IEC 61260 Terzbandfilterbank

Implementiert eine normgerechte 1/3-Oktav-Filterbank für akustische Analyse.

WICHTIG: Dies ist eine ECHTE Filterbank mit IIR-Filtern, keine spektrale Näherung.

Technische Spezifikation:
- Mittenfrequenzen nach IEC 61260-1:2014
- Butterworth IIR-Filter 6. Ordnung (Class 1 Anforderungen)
- Bandbreite: fm × (2^(1/6) - 2^(-1/6)) ≈ 0.2316 × fm

Dokumentierte Einschränkungen:
- IIR-Filter haben nichtlineares Phasenverhalten
- Gruppenlaufzeit variiert mit Frequenz (besonders an Bandgrenzen)
- Für phasenkritische Anwendungen: Zero-Phase-Filterung (filtfilt) optional

Vereinfachungen:
- Keine Kalibrierung auf akustische Referenzpegel
- Keine A/C-Gewichtung integriert
- Keine Echtzeit-Optimierung
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy import signal
from enum import Enum


class OctaveFraction(Enum):
    """Oktavbruchteil für Filterbank."""
    OCTAVE = 1
    THIRD_OCTAVE = 3
    SIXTH_OCTAVE = 6
    TWELFTH_OCTAVE = 12


# IEC 61260-1:2014 Referenz-Mittenfrequenzen für 1/3-Oktaven (in Hz)
# Basiert auf der Referenzfrequenz 1000 Hz
# fm = 1000 × 10^(n/10) für n = ..., -10, -9, ..., 9, 10, ...
IEC_61260_CENTER_FREQUENCIES = np.array([
    # Tiefe Frequenzen (oft unter 20 Hz, nicht alle hörbar)
    12.5, 16, 20,
    # Bassbereich
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    # Mittenbereich
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
    # Hohe Frequenzen
    3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
])


@dataclass
class FilterBandInfo:
    """
    Information über ein einzelnes Filterband.
    
    Dokumentiert die technischen Eigenschaften des Filters.
    """
    center_frequency: float  # Hz
    lower_frequency: float   # Hz, -3 dB Grenze
    upper_frequency: float   # Hz, -3 dB Grenze
    bandwidth: float         # Hz
    filter_order: int
    filter_type: str        # "butterworth", "bessel", etc.
    
    # Phasen- und Gruppenlaufzeit-Information
    group_delay_at_center: float  # Samples bei fc
    phase_linear: bool     # Ist die Phase linear? (Nein bei IIR)
    
    @property
    def quality_factor(self) -> float:
        """Q-Faktor des Filters."""
        return self.center_frequency / self.bandwidth


@dataclass
class ThirdOctaveBand:
    """
    Ergebnis der Filterung eines Signals durch ein Terzband.
    
    Enthält das gefilterte Signal und alle Metadaten.
    """
    center_frequency: float
    filtered_signal: np.ndarray
    sample_rate: int
    band_info: FilterBandInfo
    
    def rms(self) -> float:
        """RMS-Pegel des gefilterten Signals."""
        return np.sqrt(np.mean(self.filtered_signal ** 2))
    
    def rms_db(self, ref: float = 1.0) -> float:
        """RMS-Pegel in dB."""
        rms = self.rms()
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms / ref)
    
    def envelope(self, method: Literal["hilbert", "rms"] = "hilbert") -> np.ndarray:
        """
        Berechne Hüllkurve des gefilterten Signals.
        
        Hilbert: Analytisches Signal, präzise Hüllkurve
        RMS: Gleitender RMS, robuster aber weniger präzise
        """
        if method == "hilbert":
            analytic = signal.hilbert(self.filtered_signal)
            return np.abs(analytic)
        else:
            # Gleitender RMS mit Fenstergröße proportional zur Periodendauer
            window_size = max(int(self.sample_rate / self.center_frequency * 2), 1)
            kernel = np.ones(window_size) / window_size
            squared = self.filtered_signal ** 2
            return np.sqrt(np.convolve(squared, kernel, mode='same'))


class ThirdOctaveFilterbank:
    """
    IEC 61260 konforme 1/3-Oktav-Filterbank.
    
    Erstellt Butterworth IIR-Bandpassfilter für jede Terzbandmittenfrequenz.
    
    Verwendung:
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        bands = fb.filter_signal(audio_data)
        
        for band in bands:
            print(f"{band.center_frequency} Hz: {band.rms_db():.1f} dB")
    
    Technische Details:
    - Filter werden bei Initialisierung für alle gültigen Frequenzen erstellt
    - Frequenzen über Nyquist/2 werden automatisch ausgeschlossen
    - Filterkoeffizienten werden als Second-Order Sections (SOS) gespeichert
      für numerische Stabilität
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
        Initialisiere Filterbank.
        
        Args:
            sample_rate: Abtastrate in Hz
            filter_order: Ordnung der Butterworth-Filter (Standard: 6)
            f_min: Minimale Mittenfrequenz (Standard: 20 Hz)
            f_max: Maximale Mittenfrequenz (default: Nyquist/2.5)
            use_zero_phase: Zero-Phase-Filterung (filtfilt) für lineare Phase
        """
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.use_zero_phase = use_zero_phase
        self.nyquist = sample_rate / 2
        
        # Bestimme gültige Frequenzgrenzen
        if f_max is None:
            # Obere Bandgrenze muss unter Nyquist liegen
            # Bei 1/3-Oktave ist die obere Grenze fm × 2^(1/6) ≈ 1.122 × fm
            f_max = self.nyquist / 1.2
        
        # Filtere gültige Mittenfrequenzen
        self.center_frequencies = IEC_61260_CENTER_FREQUENCIES[
            (IEC_61260_CENTER_FREQUENCIES >= f_min) &
            (IEC_61260_CENTER_FREQUENCIES <= f_max)
        ].copy()
        
        # Erstelle Filter für jede Mittenfrequenz
        self._filters: dict[float, tuple[np.ndarray, FilterBandInfo]] = {}
        self._create_filters()
    
    def _create_filters(self) -> None:
        """Erstelle alle Bandpassfilter."""
        for fc in self.center_frequencies:
            sos, band_info = self._design_band_filter(fc)
            self._filters[fc] = (sos, band_info)
    
    def _design_band_filter(
        self,
        center_freq: float,
    ) -> tuple[np.ndarray, FilterBandInfo]:
        """
        Entwerfe Butterworth-Bandpassfilter für eine Mittenfrequenz.
        
        Bandgrenzen nach IEC 61260:
        - Untere Grenze: fc / 2^(1/6)
        - Obere Grenze: fc × 2^(1/6)
        
        Args:
            center_freq: Mittenfrequenz in Hz
            
        Returns:
            Tuple aus (SOS-Koeffizienten, Bandinformation)
        """
        # IEC 61260 Bandgrenzen für 1/3-Oktave
        factor = 2 ** (1/6)  # ≈ 1.1225
        f_low = center_freq / factor
        f_high = center_freq * factor
        
        # Normalisiere auf Nyquist-Frequenz für scipy
        low_normalized = f_low / self.nyquist
        high_normalized = f_high / self.nyquist
        
        # Sicherheitsprüfung
        if high_normalized >= 1.0:
            high_normalized = 0.99
        if low_normalized <= 0:
            low_normalized = 0.001
        
        # Entwerfe Butterworth-Filter als SOS für Stabilität
        sos = signal.butter(
            self.filter_order,
            [low_normalized, high_normalized],
            btype='bandpass',
            output='sos',
        )
        
        # Berechne Gruppenlaufzeit bei Mittenfrequenz
        w, gd = signal.group_delay(
            signal.sos2tf(sos),
            w=[2 * np.pi * center_freq / self.sample_rate],
        )
        group_delay_samples = gd[0] if len(gd) > 0 else 0
        
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
        Filtere Signal durch alle (oder ausgewählte) Terzbänder.
        
        Args:
            data: Audiosignal (1D)
            frequencies: Optional, Liste spezifischer Mittenfrequenzen
            
        Returns:
            Liste von ThirdOctaveBand-Objekten
        """
        if data.ndim != 1:
            raise ValueError("Signal muss 1D sein (Mono)")
        
        if frequencies is None:
            frequencies = list(self.center_frequencies)
        
        results = []
        for fc in frequencies:
            if fc not in self._filters:
                raise ValueError(f"Frequenz {fc} Hz nicht in Filterbank verfügbar")
            
            sos, band_info = self._filters[fc]
            
            if self.use_zero_phase:
                # Zero-Phase-Filterung: doppelte Ordnung, lineare Phase
                filtered = signal.sosfiltfilt(sos, data)
            else:
                # Standard IIR-Filterung: kausale Filterung
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
        """Filtere Signal durch einzelnes Terzband."""
        results = self.filter_signal(data, [center_frequency])
        return results[0]
    
    def compute_time_varying_levels(
        self,
        data: np.ndarray,
        time_resolution_ms: float = 10.0,
        level_type: Literal["rms", "peak", "envelope"] = "rms",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Berechne zeitabhängige Pegel für alle Terzbänder.
        
        Dies ist die Kernfunktion für Impulsantwort-Analyse.
        
        Args:
            data: Audiosignal (1D)
            time_resolution_ms: Zeitauflösung in Millisekunden
            level_type: Art der Pegelberechnung
            
        Returns:
            Tuple aus:
            - levels: Shape (num_bands, num_time_frames), Pegel pro Band und Zeit
            - times: Zeitachse in Sekunden
            - frequencies: Mittenfrequenzen
        """
        # Berechne Frame-Parameter
        samples_per_frame = int(self.sample_rate * time_resolution_ms / 1000)
        num_frames = len(data) // samples_per_frame
        
        # Filtere alle Bänder
        bands = self.filter_signal(data)
        
        # Initialisiere Ergebnis-Array
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
                    # Spitzenwert der Hüllkurve
                    env = np.abs(signal.hilbert(segment))
                    levels[i, frame] = np.max(env)
        
        # Zeitachse
        times = np.arange(num_frames) * time_resolution_ms / 1000
        
        # Frequenzen
        frequencies = np.array([b.center_frequency for b in bands])
        
        return levels, times, frequencies
    
    def get_band_info(self, center_frequency: float) -> FilterBandInfo:
        """Hole technische Information zu einem Band."""
        if center_frequency not in self._filters:
            raise ValueError(f"Frequenz {center_frequency} Hz nicht verfügbar")
        return self._filters[center_frequency][1]
    
    def get_all_band_info(self) -> list[FilterBandInfo]:
        """Hole technische Information zu allen Bändern."""
        return [self._filters[fc][1] for fc in self.center_frequencies]
    
    @property
    def num_bands(self) -> int:
        """Anzahl der Frequenzbänder."""
        return len(self.center_frequencies)
    
    def frequency_response(
        self,
        center_frequency: float,
        num_points: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Berechne Frequenzgang eines Filters.
        
        Für Dokumentation und Visualisierung des Filterverhaltens.
        
        Returns:
            Tuple aus (Frequenzen in Hz, Magnitude in dB, Phase in Grad)
        """
        if center_frequency not in self._filters:
            raise ValueError(f"Frequenz {center_frequency} Hz nicht verfügbar")
        
        sos, _ = self._filters[center_frequency]
        
        # Berechne Frequenzgang
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
    Berechne 1/3-Oktavspektrum (zeitgemittelt) eines Signals.
    
    Shortcut-Funktion für einfache Spektralanalyse.
    
    Args:
        data: Audiosignal (1D)
        sample_rate: Samplerate
        f_min: Minimale Frequenz
        f_max: Maximale Frequenz
        
    Returns:
        Tuple aus (Mittenfrequenzen, RMS-Pegel in dB)
    """
    fb = ThirdOctaveFilterbank(sample_rate, f_min=f_min, f_max=f_max)
    bands = fb.filter_signal(data)
    
    frequencies = np.array([b.center_frequency for b in bands])
    levels_db = np.array([b.rms_db() for b in bands])
    
    return frequencies, levels_db

