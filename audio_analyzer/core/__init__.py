"""
Core DSP module - fully testable without GUI dependencies.

This module contains all signal processing logic:
- Audio I/O (WAV, MP3)
- Spectral analysis
- IEC 61260 third-octave filterbank
- Impulse response computation
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

