"""
Tests für Signalverarbeitungs-Modul.
"""

import pytest
import numpy as np

from audio_analyzer.core.signal_processing import (
    resample_audio,
    downmix_to_mono,
    normalize_audio,
    compute_rms,
    compute_peak,
    compute_crest_factor,
    compute_dc_offset,
    remove_dc_offset,
    apply_window,
)


class TestResampling:
    """Tests für Resampling-Funktionen."""
    
    def test_resample_identity(self):
        """Kein Resampling bei gleicher Rate."""
        data = np.random.randn(1000)
        result = resample_audio(data, 44100, 44100)
        np.testing.assert_array_equal(result, data)
    
    def test_resample_downsample(self):
        """Downsampling 44100 → 22050."""
        data = np.random.randn(44100)
        result = resample_audio(data, 44100, 22050)
        
        # Sollte halb so viele Samples haben
        assert len(result) == 22050
    
    def test_resample_upsample(self):
        """Upsampling 22050 → 44100."""
        data = np.random.randn(22050)
        result = resample_audio(data, 22050, 44100)
        
        # Sollte doppelt so viele Samples haben
        assert len(result) == 44100
    
    def test_resample_stereo(self):
        """Resampling von Stereo-Daten."""
        data = np.random.randn(44100, 2)
        result = resample_audio(data, 44100, 22050)
        
        assert result.shape == (22050, 2)
    
    def test_resample_preserves_frequency(self):
        """Resampling erhält Frequenzinhalt."""
        # 440 Hz Sinuston
        sr_orig = 44100
        sr_new = 22050
        duration = 1.0
        
        t_orig = np.arange(int(sr_orig * duration)) / sr_orig
        signal = np.sin(2 * np.pi * 440 * t_orig)
        
        resampled = resample_audio(signal, sr_orig, sr_new)
        
        # FFT des resampelten Signals
        spectrum = np.abs(np.fft.rfft(resampled))
        freqs = np.fft.rfftfreq(len(resampled), 1/sr_new)
        
        # Peak sollte bei ~440 Hz sein
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 440) < 10  # Toleranz 10 Hz


class TestDownmix:
    """Tests für Downmix-Funktionen."""
    
    def test_downmix_average(self):
        """Test Average-Downmix."""
        left = np.ones(100)
        right = np.ones(100) * 3
        stereo = np.column_stack([left, right])
        
        mono = downmix_to_mono(stereo, method="average")
        
        np.testing.assert_array_almost_equal(mono, np.ones(100) * 2)
    
    def test_downmix_left(self):
        """Test Left-only Downmix."""
        left = np.ones(100)
        right = np.ones(100) * 3
        stereo = np.column_stack([left, right])
        
        mono = downmix_to_mono(stereo, method="left")
        
        np.testing.assert_array_equal(mono, left)
    
    def test_downmix_right(self):
        """Test Right-only Downmix."""
        left = np.ones(100)
        right = np.ones(100) * 3
        stereo = np.column_stack([left, right])
        
        mono = downmix_to_mono(stereo, method="right")
        
        np.testing.assert_array_equal(mono, right)
    
    def test_downmix_side(self):
        """Test Side-Signal (L-R)/2."""
        left = np.ones(100) * 2
        right = np.ones(100)
        stereo = np.column_stack([left, right])
        
        side = downmix_to_mono(stereo, method="side")
        
        np.testing.assert_array_almost_equal(side, np.ones(100) * 0.5)
    
    def test_downmix_mono_passthrough(self):
        """Mono-Eingabe bleibt unverändert."""
        mono = np.random.randn(100)
        result = downmix_to_mono(mono)
        
        np.testing.assert_array_equal(result, mono)


class TestNormalization:
    """Tests für Normalisierungs-Funktionen."""
    
    def test_normalize_peak(self):
        """Peak-Normalisierung."""
        data = np.array([0.25, -0.5, 0.3])
        normalized, factor = normalize_audio(data, target_peak=1.0)
        
        assert compute_peak(normalized) == pytest.approx(1.0)
        assert factor == pytest.approx(2.0)  # 0.5 → 1.0
    
    def test_normalize_zero_signal(self):
        """Nullsignal bleibt Null."""
        data = np.zeros(100)
        normalized, factor = normalize_audio(data)
        
        np.testing.assert_array_equal(normalized, data)
        assert factor == 1.0


class TestLevelMeasurement:
    """Tests für Pegelmessung."""
    
    def test_rms_sine(self):
        """RMS eines Sinustons = Peak/√2."""
        t = np.arange(44100) / 44100
        sine = np.sin(2 * np.pi * 440 * t)
        
        rms = compute_rms(sine)
        expected = 1.0 / np.sqrt(2)
        
        assert rms == pytest.approx(expected, rel=0.01)
    
    def test_rms_db(self):
        """RMS in dB."""
        data = np.ones(100) * 0.1  # -20 dB
        rms_db = compute_rms(data, as_db=True)
        
        assert rms_db == pytest.approx(-20.0, rel=0.01)
    
    def test_peak(self):
        """Peak-Wert."""
        data = np.array([0.3, -0.8, 0.5])
        peak = compute_peak(data)
        
        assert peak == 0.8
    
    def test_crest_factor_sine(self):
        """Crest Factor Sinuston ≈ √2."""
        t = np.arange(44100) / 44100
        sine = np.sin(2 * np.pi * 440 * t)
        
        cf = compute_crest_factor(sine)
        
        assert cf == pytest.approx(np.sqrt(2), rel=0.01)


class TestDCOffset:
    """Tests für DC-Offset-Handling."""
    
    def test_dc_offset_detection(self):
        """DC-Offset erkennen."""
        data = np.ones(100) * 0.1 + np.random.randn(100) * 0.01
        dc = compute_dc_offset(data)
        
        assert dc == pytest.approx(0.1, rel=0.1)
    
    def test_dc_offset_removal(self):
        """DC-Offset entfernen."""
        data = np.ones(100) * 0.5
        result = remove_dc_offset(data)
        
        assert compute_dc_offset(result) == pytest.approx(0.0, abs=1e-10)


class TestWindowing:
    """Tests für Fensterfunktionen."""
    
    def test_window_shapes(self):
        """Alle Fenstertypen produzieren korrekte Länge."""
        for window in ["hann", "hamming", "blackman", "kaiser", "rectangular"]:
            result = apply_window(np.ones(1024), window)
            assert len(result) == 1024
    
    def test_rectangular_window(self):
        """Rechteckfenster verändert nichts."""
        data = np.ones(100)
        result = apply_window(data, "rectangular")
        
        np.testing.assert_array_equal(result, data)
    
    def test_hann_window_zeros_at_edges(self):
        """Hann-Fenster hat Nullen an den Rändern."""
        data = np.ones(100)
        result = apply_window(data, "hann")
        
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[-1] == pytest.approx(0.0, abs=1e-10)

