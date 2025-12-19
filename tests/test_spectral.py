"""
Tests für Spektralanalyse-Modul.
"""

import pytest
import numpy as np

from audio_analyzer.core.spectral import (
    SpectrogramConfig,
    SpectrogramResult,
    compute_spectrogram,
    compute_power_spectrum,
    frequency_to_log_scale,
)


class TestSpectrogramConfig:
    """Tests für Spektrogramm-Konfiguration."""
    
    def test_default_config(self):
        """Standard-Konfiguration ist gültig."""
        config = SpectrogramConfig()
        
        assert config.fft_size == 2048
        assert config.window == "hann"
        assert config.overlap_percent == 75.0
    
    def test_hop_from_overlap(self):
        """Hop-Size wird aus Überlappung berechnet."""
        config = SpectrogramConfig(fft_size=1024, overlap_percent=50.0)
        
        assert config.hop_size == 512
    
    def test_explicit_hop(self):
        """Explizite Hop-Size überschreibt Überlappung."""
        config = SpectrogramConfig(fft_size=1024, hop_size=256)
        
        assert config.hop_size == 256
    
    def test_resolution_calculation(self):
        """Auflösungsberechnung ist korrekt."""
        config = SpectrogramConfig(fft_size=2048, hop_size=512)
        
        # Bei 44100 Hz
        freq_res = config.frequency_resolution(44100)
        time_res = config.time_resolution(44100)
        
        assert freq_res == pytest.approx(44100 / 2048, rel=0.01)
        assert time_res == pytest.approx(512 / 44100, rel=0.01)
    
    def test_invalid_fft_size(self):
        """Ungültige FFT-Größe wird abgelehnt."""
        with pytest.raises(ValueError):
            SpectrogramConfig(fft_size=16)  # Zu klein
    
    def test_invalid_hop_size(self):
        """Ungültige Hop-Size wird abgelehnt."""
        with pytest.raises(ValueError):
            SpectrogramConfig(fft_size=1024, hop_size=2048)  # Größer als FFT


class TestSpectrogram:
    """Tests für Spektrogramm-Berechnung."""
    
    def test_spectrogram_shape(self):
        """Spektrogramm hat korrekte Form."""
        sr = 44100
        data = np.random.randn(sr)  # 1 Sekunde
        
        config = SpectrogramConfig(fft_size=1024, hop_size=256)
        result = compute_spectrogram(data, sr, config)
        
        # Frequenzbins: fft_size/2 + 1 (einseitig)
        assert result.magnitude.shape[0] == 513
        
        # Zeitframes: abhängig von Padding
        assert result.magnitude.shape[1] > 0
    
    def test_frequency_axis(self):
        """Frequenzachse ist korrekt."""
        sr = 44100
        data = np.random.randn(sr)
        
        result = compute_spectrogram(data, sr)
        
        # Sollte bei 0 Hz starten
        assert result.frequencies[0] == 0
        
        # Sollte bei Nyquist enden
        assert result.frequencies[-1] == pytest.approx(sr / 2, rel=0.01)
    
    def test_sine_peak_detection(self):
        """Sinuston erscheint bei korrekter Frequenz."""
        sr = 44100
        freq = 1000  # 1 kHz
        
        t = np.arange(sr) / sr
        sine = np.sin(2 * np.pi * freq * t)
        
        config = SpectrogramConfig(fft_size=4096)
        result = compute_spectrogram(sine, sr, config)
        
        # Mittlere Magnitude über Zeit
        mean_mag = np.mean(result.magnitude, axis=1)
        
        # Peak sollte bei ~1000 Hz sein
        peak_idx = np.argmax(mean_mag)
        peak_freq = result.frequencies[peak_idx]
        
        assert abs(peak_freq - freq) < 20  # Toleranz 20 Hz
    
    def test_magnitude_db_conversion(self):
        """dB-Konvertierung funktioniert."""
        sr = 44100
        data = np.random.randn(sr) * 0.1  # -20 dB Signal
        
        result = compute_spectrogram(data, sr)
        mag_db = result.magnitude_db()
        
        # Sollte negative Werte haben (unter 0 dB)
        assert np.mean(mag_db) < 0
    
    def test_phase_output(self):
        """Phase wird berechnet."""
        sr = 44100
        data = np.random.randn(sr)
        
        result = compute_spectrogram(data, sr)
        
        # Phase sollte im Bereich [-π, π] sein
        assert np.all(result.phase >= -np.pi)
        assert np.all(result.phase <= np.pi)


class TestPowerSpectrum:
    """Tests für Leistungsspektrum."""
    
    def test_power_spectrum_shape(self):
        """Leistungsspektrum hat korrekte Form."""
        sr = 44100
        data = np.random.randn(sr)
        
        freqs, psd = compute_power_spectrum(data, sr)
        
        assert len(freqs) == len(psd)
        assert len(freqs) > 0
    
    def test_sine_power_spectrum(self):
        """Sinuston hat Peak im Leistungsspektrum."""
        sr = 44100
        freq = 440
        
        t = np.arange(sr * 5) / sr
        sine = np.sin(2 * np.pi * freq * t)
        
        freqs, psd = compute_power_spectrum(sine, sr)
        
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        
        assert abs(peak_freq - freq) < 10


class TestLogFrequencyScale:
    """Tests für logarithmische Frequenzskala."""
    
    def test_log_scale_shape(self):
        """Logarithmische Skala hat korrekte Form."""
        mag = np.random.rand(1025, 100)  # frequencies × time
        freqs = np.linspace(0, 22050, 1025)
        
        log_mag, log_freqs = frequency_to_log_scale(mag, freqs, num_bins=256)
        
        assert log_mag.shape == (256, 100)
        assert len(log_freqs) == 256
    
    def test_log_scale_frequency_range(self):
        """Frequenzbereich ist korrekt."""
        mag = np.random.rand(1025, 100)
        freqs = np.linspace(0, 22050, 1025)
        
        log_mag, log_freqs = frequency_to_log_scale(
            mag, freqs,
            f_min=20,
            f_max=20000,
        )
        
        assert log_freqs[0] == pytest.approx(20, rel=0.01)
        assert log_freqs[-1] == pytest.approx(20000, rel=0.01)
    
    def test_log_scale_distribution(self):
        """Frequenzen sind logarithmisch verteilt."""
        mag = np.random.rand(1025, 100)
        freqs = np.linspace(0, 22050, 1025)
        
        _, log_freqs = frequency_to_log_scale(mag, freqs, num_bins=100)
        
        # Verhältnis benachbarter Frequenzen sollte konstant sein
        ratios = log_freqs[1:] / log_freqs[:-1]
        
        assert np.std(ratios) < 0.01  # Sehr kleine Varianz

