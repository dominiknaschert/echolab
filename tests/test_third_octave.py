"""
Tests für IEC 61260 Terzbandfilterbank.

Diese Tests verifizieren die korrekte Implementierung der Filterbank.
"""

import pytest
import numpy as np

from audio_analyzer.core.third_octave import (
    ThirdOctaveFilterbank,
    ThirdOctaveBand,
    IEC_61260_CENTER_FREQUENCIES,
    compute_third_octave_spectrum,
)


class TestCenterFrequencies:
    """Tests für IEC 61260 Mittenfrequenzen."""
    
    def test_standard_frequencies_present(self):
        """Standardfrequenzen sind enthalten."""
        standard = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        
        for freq in standard:
            assert freq in IEC_61260_CENTER_FREQUENCIES
    
    def test_frequency_count(self):
        """Korrekte Anzahl Terzbänder."""
        # Von 12.5 Hz bis 20 kHz: ~31 Bänder
        assert len(IEC_61260_CENTER_FREQUENCIES) >= 30
    
    def test_frequency_ratio(self):
        """Verhältnis benachbarter Frequenzen ≈ 2^(1/3)."""
        expected_ratio = 2 ** (1/3)  # ≈ 1.26
        
        for i in range(1, len(IEC_61260_CENTER_FREQUENCIES)):
            ratio = IEC_61260_CENTER_FREQUENCIES[i] / IEC_61260_CENTER_FREQUENCIES[i-1]
            assert ratio == pytest.approx(expected_ratio, rel=0.05)


class TestFilterbankCreation:
    """Tests für Filterbank-Erstellung."""
    
    def test_filterbank_creation(self):
        """Filterbank wird korrekt erstellt."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        assert fb.sample_rate == 44100
        assert fb.num_bands > 0
    
    def test_nyquist_limit(self):
        """Keine Filter über Nyquist-Frequenz."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        nyquist = 44100 / 2
        for fc in fb.center_frequencies:
            # Obere Bandgrenze muss unter Nyquist sein
            f_high = fc * (2 ** (1/6))
            assert f_high < nyquist
    
    def test_frequency_range(self):
        """Frequenzbereich-Einschränkung funktioniert."""
        fb = ThirdOctaveFilterbank(
            sample_rate=44100,
            f_min=100,
            f_max=4000,
        )
        
        assert fb.center_frequencies[0] >= 100
        assert fb.center_frequencies[-1] <= 4000
    
    def test_low_sample_rate(self):
        """Niedrige Samplerate hat weniger Bänder."""
        fb_high = ThirdOctaveFilterbank(sample_rate=44100)
        fb_low = ThirdOctaveFilterbank(sample_rate=8000)
        
        assert fb_low.num_bands < fb_high.num_bands


class TestFiltering:
    """Tests für Filterung."""
    
    def test_filter_sine_at_center(self):
        """Sinuston bei Mittenfrequenz passiert Filter."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        # 1 kHz ist eine Standard-Mittenfrequenz
        fc = 1000
        t = np.arange(44100) / 44100
        sine = np.sin(2 * np.pi * fc * t)
        
        band = fb.filter_single_band(sine, fc)
        
        # RMS des gefilterten Signals sollte nahe am Original sein
        original_rms = np.sqrt(np.mean(sine ** 2))
        filtered_rms = band.rms()
        
        # Erwarte mindestens 80% Durchlass
        assert filtered_rms > original_rms * 0.8
    
    def test_filter_sine_outside_band(self):
        """Sinuston außerhalb des Bandes wird gedämpft."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        # 1 kHz Band, aber 4 kHz Sinus
        fc_band = 1000
        fc_signal = 4000
        
        t = np.arange(44100) / 44100
        sine = np.sin(2 * np.pi * fc_signal * t)
        
        band = fb.filter_single_band(sine, fc_band)
        
        original_rms = np.sqrt(np.mean(sine ** 2))
        filtered_rms = band.rms()
        
        # Erwarte starke Dämpfung (< 1% des Originals)
        assert filtered_rms < original_rms * 0.01
    
    def test_filter_all_bands(self):
        """Alle Bänder werden berechnet."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        noise = np.random.randn(44100)
        bands = fb.filter_signal(noise)
        
        assert len(bands) == fb.num_bands
        
        for band in bands:
            assert isinstance(band, ThirdOctaveBand)
            assert len(band.filtered_signal) == len(noise)
    
    def test_zero_phase_filtering(self):
        """Zero-Phase-Filterung hat keine Gruppenlaufzeit."""
        fb_causal = ThirdOctaveFilterbank(sample_rate=44100, use_zero_phase=False)
        fb_zero = ThirdOctaveFilterbank(sample_rate=44100, use_zero_phase=True)
        
        # Impuls
        impulse = np.zeros(4410)
        impulse[2205] = 1.0  # Impuls in der Mitte
        
        fc = 1000
        band_causal = fb_causal.filter_single_band(impulse, fc)
        band_zero = fb_zero.filter_single_band(impulse, fc)
        
        # Peak-Position
        peak_causal = np.argmax(np.abs(band_causal.filtered_signal))
        peak_zero = np.argmax(np.abs(band_zero.filtered_signal))
        
        # Zero-Phase sollte näher am Original-Peak (2205) sein
        assert abs(peak_zero - 2205) < abs(peak_causal - 2205)


class TestBandInfo:
    """Tests für Filter-Informationen."""
    
    def test_band_info_completeness(self):
        """Alle Bandinformationen sind vorhanden."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        for fc in fb.center_frequencies:
            info = fb.get_band_info(fc)
            
            assert info.center_frequency == fc
            assert info.lower_frequency < fc
            assert info.upper_frequency > fc
            assert info.bandwidth > 0
            assert info.filter_order > 0
            assert info.quality_factor > 0
    
    def test_bandwidth_calculation(self):
        """Bandbreite entspricht 1/3-Oktave."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        fc = 1000
        info = fb.get_band_info(fc)
        
        # Theoretische Bandbreite für 1/3-Oktave
        expected_bw = fc * (2**(1/6) - 2**(-1/6))  # ≈ 0.2316 × fc
        
        assert info.bandwidth == pytest.approx(expected_bw, rel=0.01)


class TestFrequencyResponse:
    """Tests für Frequenzgang."""
    
    def test_frequency_response_shape(self):
        """Frequenzgang hat erwartete Form."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        freqs, mag, phase = fb.frequency_response(1000)
        
        assert len(freqs) == len(mag) == len(phase)
        assert len(freqs) > 0
    
    def test_passband_gain(self):
        """Verstärkung im Durchlassbereich ≈ 0 dB."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        freqs, mag, _ = fb.frequency_response(1000, num_points=10000)
        
        # Finde Magnitude bei Mittenfrequenz
        fc_idx = np.argmin(np.abs(freqs - 1000))
        mag_at_fc = mag[fc_idx]
        
        # Sollte nahe 0 dB sein
        assert mag_at_fc == pytest.approx(0.0, abs=1.0)
    
    def test_stopband_attenuation(self):
        """Dämpfung im Sperrbereich."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        freqs, mag, _ = fb.frequency_response(1000, num_points=10000)
        
        # Bei 4 kHz (2 Oktaven entfernt) sollte starke Dämpfung sein
        idx_4k = np.argmin(np.abs(freqs - 4000))
        
        assert mag[idx_4k] < -30  # Mindestens 30 dB Dämpfung


class TestTimeVaryingLevels:
    """Tests für zeitabhängige Pegelberechnung."""
    
    def test_time_varying_output_shape(self):
        """Ausgabeform ist korrekt."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        
        # 1 Sekunde Audio, 10 ms Zeitauflösung = 100 Frames
        data = np.random.randn(44100)
        levels, times, freqs = fb.compute_time_varying_levels(
            data,
            time_resolution_ms=10.0,
        )
        
        assert levels.shape[0] == fb.num_bands
        assert levels.shape[1] == 100
        assert len(times) == 100
        assert len(freqs) == fb.num_bands
    
    def test_time_varying_level_types(self):
        """Verschiedene Pegeltypen funktionieren."""
        fb = ThirdOctaveFilterbank(sample_rate=44100)
        data = np.random.randn(44100)
        
        for level_type in ["rms", "peak", "envelope"]:
            levels, _, _ = fb.compute_time_varying_levels(
                data,
                level_type=level_type,
            )
            
            assert not np.any(np.isnan(levels))


class TestThirdOctaveSpectrum:
    """Tests für Spektrum-Berechnung."""
    
    def test_spectrum_computation(self):
        """Spektrum wird korrekt berechnet."""
        # Weißes Rauschen sollte flaches Spektrum haben
        noise = np.random.randn(44100 * 10)  # 10 Sekunden
        
        freqs, levels_db = compute_third_octave_spectrum(noise, 44100)
        
        # Alle Pegel sollten ähnlich sein (±10 dB)
        assert np.std(levels_db) < 10

