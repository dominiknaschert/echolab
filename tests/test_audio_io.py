"""
Tests für Audio I/O Modul.

Testet Laden und Speichern von Audiodateien ohne GUI.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from audio_analyzer.core.audio_io import (
    AudioFile,
    load_audio,
    save_audio,
)


class TestAudioFile:
    """Tests für AudioFile Dataclass."""
    
    def test_mono_audio_file(self):
        """Test AudioFile mit Mono-Daten."""
        data = np.random.randn(44100).astype(np.float64)
        audio = AudioFile(
            data=data,
            sample_rate=44100,
            channels=1,
            duration_seconds=1.0,
            num_samples=44100,
            file_path=Path("test.wav"),
        )
        
        assert audio.channels == 1
        assert audio.duration_seconds == 1.0
        assert len(audio.get_channel(0)) == 44100
    
    def test_stereo_audio_file(self):
        """Test AudioFile mit Stereo-Daten."""
        data = np.random.randn(44100, 2).astype(np.float64)
        audio = AudioFile(
            data=data,
            sample_rate=44100,
            channels=2,
            duration_seconds=1.0,
            num_samples=44100,
            file_path=Path("test.wav"),
        )
        
        assert audio.channels == 2
        assert audio.get_channel(0).shape == (44100,)
        assert audio.get_channel(1).shape == (44100,)
    
    def test_time_sample_conversion(self):
        """Test Zeit-Sample-Konvertierung."""
        data = np.zeros(88200)
        audio = AudioFile(
            data=data,
            sample_rate=44100,
            channels=1,
            duration_seconds=2.0,
            num_samples=88200,
            file_path=Path("test.wav"),
        )
        
        # 1 Sekunde = 44100 Samples
        assert audio.time_to_sample(1.0) == 44100
        assert audio.sample_to_time(44100) == 1.0
        
        # Randwerte
        assert audio.time_to_sample(0) == 0
        assert audio.time_to_sample(10.0) == 88199  # Clamped
    
    def test_get_time_range(self):
        """Test Zeitbereich-Extraktion."""
        data = np.arange(44100, dtype=np.float64)
        audio = AudioFile(
            data=data,
            sample_rate=44100,
            channels=1,
            duration_seconds=1.0,
            num_samples=44100,
            file_path=Path("test.wav"),
        )
        
        segment = audio.get_time_range(0, 1000)
        assert len(segment) == 1000
        assert segment[0] == 0
        assert segment[999] == 999
        
        # Original darf nicht verändert werden
        segment[0] = 999
        assert audio.data[0] == 0


class TestSaveLoadAudio:
    """Tests für Speichern und Laden von Audiodateien."""
    
    def test_save_load_wav_mono(self):
        """Test WAV speichern und laden (Mono)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_mono.wav"
            
            # Erstelle Testdaten
            original = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
            
            # Speichern
            save_audio(original, filepath, sample_rate=44100)
            
            # Laden
            loaded = load_audio(filepath)
            
            assert loaded.sample_rate == 44100
            assert loaded.channels == 1
            assert loaded.num_samples == 44100
            
            # Daten sollten nahezu identisch sein (PCM-Quantisierung)
            np.testing.assert_allclose(loaded.data, original, atol=1e-4)
    
    def test_save_load_wav_stereo(self):
        """Test WAV speichern und laden (Stereo)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_stereo.wav"
            
            # Erstelle Stereo-Testdaten
            left = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
            right = np.sin(2 * np.pi * 880 * np.arange(44100) / 44100)
            original = np.column_stack([left, right])
            
            # Speichern
            save_audio(original, filepath, sample_rate=44100)
            
            # Laden
            loaded = load_audio(filepath)
            
            assert loaded.channels == 2
            assert loaded.data.shape == (44100, 2)
    
    def test_clipping_warning(self):
        """Test Warnung bei Clipping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_clip.wav"
            
            # Daten über 1.0
            data = np.array([0.5, 1.5, -1.5, 0.8])
            
            # Sollte Warnung erzeugen
            with pytest.warns(UserWarning, match="Clipping"):
                save_audio(data, filepath, sample_rate=44100)


class TestFileNotFound:
    """Tests für Fehlerbehandlung."""
    
    def test_file_not_found(self):
        """Test FileNotFoundError bei nicht existierender Datei."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")
    
    def test_unsupported_format(self):
        """Test ValueError bei nicht unterstütztem Format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
            with pytest.raises(ValueError, match="Nicht unterstütztes Format"):
                load_audio(f.name)

