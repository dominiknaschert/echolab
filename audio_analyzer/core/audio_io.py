"""
Audio I/O Module

Lädt und speichert Audiodateien (WAV, MP3) ohne implizite Signalmanipulation.
Der Nutzer entscheidet bewusst über Downmix und Resampling.

Technische Annahmen:
- WAV-Dateien werden mit soundfile geladen (hohe Präzision, keine Konvertierung)
- MP3-Dateien werden mit librosa/audioread dekodiert
- Alle Audiodaten werden als float64 numpy-Arrays zurückgegeben (Bereich -1.0 bis 1.0)
- Kanalreihenfolge bei Stereo: [links, rechts] als (samples, 2) Array
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import soundfile as sf


@dataclass
class AudioFile:
    """
    Repräsentiert eine geladene Audiodatei mit allen Metadaten.
    
    Das Audiosignal wird NICHT automatisch verändert.
    Downmix/Resampling erfolgt nur auf explizite Nutzeranweisung.
    
    Attributes:
        data: Audiodaten als numpy array, Shape: (samples,) oder (samples, channels)
        sample_rate: Original-Samplerate der Datei
        channels: Anzahl der Kanäle (1=Mono, 2=Stereo)
        duration_seconds: Dauer in Sekunden
        num_samples: Anzahl der Samples pro Kanal
        file_path: Pfad zur Quelldatei
        format_info: Formatinformationen (Subtype, Endianness)
        bit_depth: Bit-Tiefe des Originals (falls bekannt)
    """
    data: np.ndarray
    sample_rate: int
    channels: int
    duration_seconds: float
    num_samples: int
    file_path: Path
    format_info: dict = field(default_factory=dict)
    bit_depth: Optional[int] = None
    
    def __post_init__(self):
        """Validiere Datenintegrität."""
        if self.data.ndim == 1:
            assert self.channels == 1, "1D-Array muss Mono sein"
        elif self.data.ndim == 2:
            assert self.data.shape[1] == self.channels, "Kanalanzahl stimmt nicht"
        else:
            raise ValueError("Audio-Array muss 1D oder 2D sein")
    
    def get_channel(self, channel: int) -> np.ndarray:
        """
        Extrahiere einen einzelnen Kanal.
        
        Args:
            channel: 0 für links/Mono, 1 für rechts
            
        Returns:
            1D numpy array mit den Samples des Kanals
        """
        if self.channels == 1:
            if channel != 0:
                raise ValueError("Mono-Datei hat nur Kanal 0")
            return self.data if self.data.ndim == 1 else self.data[:, 0]
        else:
            if channel not in (0, 1):
                raise ValueError("Stereo unterstützt nur Kanäle 0 und 1")
            return self.data[:, channel]
    
    def time_to_sample(self, time_seconds: float) -> int:
        """Konvertiere Zeit in Sekunden zu Sample-Index."""
        sample = int(time_seconds * self.sample_rate)
        return max(0, min(sample, self.num_samples - 1))
    
    def sample_to_time(self, sample: int) -> float:
        """Konvertiere Sample-Index zu Zeit in Sekunden."""
        return sample / self.sample_rate
    
    def get_time_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        """
        Extrahiere einen Zeitbereich (nicht-destruktiv).
        
        Returns:
            Kopie der Audiodaten im angegebenen Bereich
        """
        start = max(0, start_sample)
        end = min(end_sample, self.num_samples)
        
        if self.data.ndim == 1:
            return self.data[start:end].copy()
        else:
            return self.data[start:end, :].copy()


def load_audio(file_path: str | Path) -> AudioFile:
    """
    Lade eine Audiodatei ohne implizite Konvertierung.
    
    Unterstützte Formate:
    - WAV (alle gängigen Subtypes: PCM_16, PCM_24, PCM_32, FLOAT)
    - MP3 (via librosa/audioread)
    
    Die Audiodaten werden als float64 im Bereich [-1.0, 1.0] zurückgegeben.
    KEINE automatische Konvertierung von Samplerate oder Kanalanzahl.
    
    Args:
        file_path: Pfad zur Audiodatei
        
    Returns:
        AudioFile-Objekt mit allen Metadaten
        
    Raises:
        FileNotFoundError: Datei existiert nicht
        ValueError: Nicht unterstütztes Format
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".wav":
        return _load_wav(path)
    elif suffix == ".mp3":
        return _load_mp3(path)
    else:
        raise ValueError(f"Nicht unterstütztes Format: {suffix}")


def _load_wav(path: Path) -> AudioFile:
    """
    Lade WAV-Datei mit soundfile.
    
    soundfile verwendet libsndfile und liefert präzise Ergebnisse
    ohne ungewollte Konvertierungen.
    """
    # Lese Audio und Samplerate
    data, sample_rate = sf.read(path, dtype='float64', always_2d=False)
    
    # Lese erweiterte Metadaten
    info = sf.info(path)
    
    # Bestimme Bit-Tiefe aus Subtype
    bit_depth = _extract_bit_depth(info.subtype)
    
    # Bestimme Kanalanzahl
    if data.ndim == 1:
        channels = 1
        num_samples = len(data)
    else:
        num_samples, channels = data.shape
    
    format_info = {
        "format": info.format,
        "subtype": info.subtype,
        "endian": info.endian,
        "sections": info.sections,
    }
    
    return AudioFile(
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=num_samples / sample_rate,
        num_samples=num_samples,
        file_path=path,
        format_info=format_info,
        bit_depth=bit_depth,
    )


def _load_mp3(path: Path) -> AudioFile:
    """
    Lade MP3-Datei mit pydub.
    
    Benötigt ffmpeg im System PATH für die Dekodierung.
    Falls ffmpeg nicht verfügbar ist, wird ein Fehler ausgegeben.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub ist nicht installiert. Bitte mit 'pip install pydub' installieren.")
    
    try:
        audio = AudioSegment.from_mp3(path)
    except Exception as e:
        raise RuntimeError(
            f"MP3 konnte nicht geladen werden: {e}\n"
            "Stellen Sie sicher, dass ffmpeg installiert ist:\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    
    # Extrahiere Metadaten
    sample_rate = audio.frame_rate
    channels = audio.channels
    
    # Konvertiere zu numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalisiere auf float64 im Bereich [-1, 1]
    # pydub gibt int16 oder int32 zurück
    if audio.sample_width == 2:  # 16-bit
        samples = samples.astype(np.float64) / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples = samples.astype(np.float64) / 2147483648.0
    else:  # 8-bit
        samples = (samples.astype(np.float64) - 128) / 128.0
    
    # Reshape für Stereo
    if channels == 2:
        samples = samples.reshape(-1, 2)
        num_samples = samples.shape[0]
    else:
        num_samples = len(samples)
    
    format_info = {
        "format": "MP3",
        "subtype": "MPEG Layer 3",
        "note": "MP3 ist verlustbehaftet, Originaldaten nicht rekonstruierbar",
    }
    
    return AudioFile(
        data=samples,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=num_samples / sample_rate,
        num_samples=num_samples,
        file_path=path,
        format_info=format_info,
        bit_depth=None,  # MP3 hat keine feste Bit-Tiefe
    )


def save_audio(
    data: np.ndarray,
    file_path: str | Path,
    sample_rate: int,
    subtype: Literal["PCM_16", "PCM_24", "PCM_32", "FLOAT"] = "PCM_24",
) -> None:
    """
    Speichere Audiodaten als WAV-Datei.
    
    Args:
        data: Audiodaten als numpy array (float64, Bereich -1.0 bis 1.0)
        file_path: Zielpfad
        sample_rate: Samplerate
        subtype: WAV-Subtype für Quantisierung
        
    Raises:
        ValueError: Ungültige Daten oder Parameter
    """
    path = Path(file_path)
    
    # Validierung
    if data.ndim > 2:
        raise ValueError("Audio muss 1D oder 2D sein")
    
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Audiodaten müssen float sein")
    
    # Clipping-Warnung
    if np.any(np.abs(data) > 1.0):
        import warnings
        warnings.warn(
            "Audiodaten überschreiten [-1.0, 1.0]. Clipping wird angewendet.",
            UserWarning
        )
        data = np.clip(data, -1.0, 1.0)
    
    sf.write(path, data, sample_rate, subtype=subtype)


def _extract_bit_depth(subtype: str) -> Optional[int]:
    """Extrahiere Bit-Tiefe aus soundfile Subtype-String."""
    bit_depth_map = {
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
        "PCM_S8": 8,
        "PCM_U8": 8,
    }
    return bit_depth_map.get(subtype)

