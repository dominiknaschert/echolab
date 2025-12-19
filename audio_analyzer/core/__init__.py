"""
Core DSP module - vollständig ohne GUI-Abhängigkeiten testbar.

Dieses Modul enthält die gesamte Signalverarbeitungslogik:
- Audio I/O (WAV, MP3)
- Spektralanalyse
- IEC 61260 Terzbandfilterbank
- Impulsantwort-Berechnung
"""

from .audio_io import AudioFile, load_audio, save_audio
from .signal_processing import (
    resample_audio,
    downmix_to_mono,
    normalize_audio,
    compute_rms,
    compute_peak,
)
from .spectral import compute_spectrogram, SpectrogramConfig
from .third_octave import ThirdOctaveFilterbank, IEC_61260_CENTER_FREQUENCIES

__all__ = [
    "AudioFile",
    "load_audio", 
    "save_audio",
    "resample_audio",
    "downmix_to_mono",
    "normalize_audio",
    "compute_rms",
    "compute_peak",
    "compute_spectrogram",
    "SpectrogramConfig",
    "ThirdOctaveFilterbank",
    "IEC_61260_CENTER_FREQUENCIES",
]

