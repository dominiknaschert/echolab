"""
Room Analysis - Core structures, wrapper and algorithm.
Based on Schanda/Hoffbauer/Lachenmayr (DAGA 2023) and the two-line regression.
See paper in `Literatur/000213.pdf`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
from scipy import signal


__all__ = [
    "PeakInfo",
    "FlutterEchoResult",
    "FlutterDetectionConfig",
    "DecayCurveResult",
    "analyze_flutter",
    "compute_decay_curve",
    "FlutterEchoDetector",
    "SOUND_SPEED",
    "distance_to_frequency",
    "frequency_to_distance",
    "distance_to_samples",
    "samples_to_distance",
    "schroeder_integration",
    "_butter_bandpass",
    "_butter_lowpass",
]
SOUND_SPEED = 343.0  # m/s bei 20°C


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PeakInfo:
    """
    Information about a single peak in the distogram.
    """
    distance_m: float              # Distance in meters
    amplitude: float               # Peak amplitude (arbitrary units)
    repetition_frequency_hz: float # Repetition frequency f_rep
    is_main: bool = False          # True for main peak


@dataclass
class FlutterEchoResult:
    """
    Result of flutter echo detection for the GUI.
    """
    distances: np.ndarray          # X-axis: distances in meters
    amplitudes: np.ndarray         # Y-axis: amplitudes (arbitrary units)
    peaks: List[PeakInfo]          # All detected peaks
    
    main_distance_m: float         # Main distance
    distance_uncertainty_m: float  # Accuracy
    repetition_frequency_hz: float # f_rep
    relative_amplitude: float      # Peak amplitude
    flutter_tonality_hz: float     # Center frequency of analysis (e.g. 1000 Hz)
    severity: str                  # Audibility ("not audible", etc.)
    detected: bool                 # Detection success
    rt60_s: float = 0.0            # Reverberation time RT60 (based on trend line)
    
    # Debug/Plot data for pipeline details
    t: Optional[np.ndarray] = None
    l_ir: Optional[np.ndarray] = None
    l_trend: Optional[np.ndarray] = None
    rir_raw: Optional[np.ndarray] = None
    rir_bp: Optional[np.ndarray] = None
    l_fe: Optional[np.ndarray] = None
    acf: Optional[np.ndarray] = None
    t_intersect: float = 0.0
    noise_level: float = 0.0
    p_decay: Optional[np.ndarray] = None
    t_start_analysis: float = 0.2  # Actually used start for analysis time windowing
    t_end_analysis: float = 0.0    # Actually used end for analysis time windowing
    
    # Backward compatibility
    peak_distances: List[float] = None
    peak_amplitudes: List[float] = None


@dataclass
class FlutterDetectionConfig:
    """Configuration for the detection algorithm."""
    sample_rate: int = 48_000
    band_center_hz: float = 1_000.0  # Octave band as in paper
    band_q: float = 1 / np.log(2)    # corresponds to ~1 octave (BW = fc / Q)
    lp_cut_hz: float = 200.0         # Low-pass for envelope
    t_start_fit: float = 0.2         # Ignore first 200ms (direct sound)
    method: Literal["regression", "schroeder"] = "regression"
    min_distance_m: float = 0.5
    max_distance_m: float = 20.0  # Maximum distance for peak detection
    peak_rel_height: float = 0.1     # Peaks: relative threshold to max
    peak_prominence: float = 0.05    # relative prominence
    fft_zero_padding_factor: int = 4  # Zero-padding for better frequency resolution (4x = 4x more points)
    # Manual override values for RT60 correction
    t_start_fit_override: Optional[float] = None  # Manual start point for regression
    t_end_fit_override: Optional[float] = None    # Manual end point for regression
    noise_level_override: Optional[float] = None  # Manual noise floor (dB)
    manual_slope: Optional[float] = None          # Manual slope (dB/s)
    manual_intercept: Optional[float] = None      # Manual Y-axis intercept (dB)


@dataclass
class DecayCurveResult:
    """Result of level progression calculation (before peak detection)."""
    t: np.ndarray               # Time axis in seconds
    l_ir: np.ndarray            # Level progression in dB
    l_trend: np.ndarray         # Trend line in dB
    rt60_s: float               # Calculated RT60
    t_start_fit: float          # Start of regression
    t_end_fit: float            # End of regression
    t_intersect: float          # Intersection with noise floor
    noise_level: float          # Noise floor in dB
    p_decay: Optional[np.ndarray]  # Polynomial coefficients of regression
    rir_bp: np.ndarray          # Bandpass-filtered signal
    sample_rate: int            # Sample rate


# ============================================================
# CORE ALGORITHM (SIGNAL PROCESSING)
# ============================================================

def _butter_bandpass(fc: float, q: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    bw = fc / q
    low = max(1e-6, (fc - bw / 2) / (fs / 2))
    high = min(0.999, (fc + bw / 2) / (fs / 2))
    return signal.butter(order, [low, high], btype="bandpass")


def _butter_lowpass(fc: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    wn = min(0.999, fc / (fs / 2))
    return signal.butter(order, wn, btype="lowpass")


def schroeder_integration(env: np.ndarray) -> np.ndarray:
    """Backward integration according to Schroeder on the energy time series."""
    rev_cum = np.cumsum(env[::-1])
    sch = rev_cum[::-1]
    if np.max(sch) > 0:
        sch = sch / np.max(sch) * np.max(env)
    return sch


def compute_decay_curve(
    impulse_response: np.ndarray,
    config: Optional[FlutterDetectionConfig] = None,
) -> DecayCurveResult:
    """
    Calculates only the level progression and initial trend line.
    Called before the RT60 correction dialog.
    """
    cfg = config or FlutterDetectionConfig()
    fs = cfg.sample_rate
    rir = np.asarray(impulse_response, dtype=float)
    t = np.arange(len(rir)) / fs

    # 1) Bandpass
    b_bp, a_bp = _butter_bandpass(cfg.band_center_hz, cfg.band_q, fs, order=4)
    rir_bp = signal.filtfilt(b_bp, a_bp, rir)

    # 2) Level-time representation (envelope in dB)
    window_len = int(0.002 * fs)
    energy = rir_bp**2
    smoothed_energy = np.convolve(energy, np.ones(window_len)/window_len, mode='same')
    l_ir = 10 * np.log10(smoothed_energy + 1e-12)

    # 3) Initial regression
    t_start = cfg.t_start_fit_override if cfg.t_start_fit_override is not None else cfg.t_start_fit
    t_end_auto = min(t[-1] * 0.8, 1.2)
    t_end = cfg.t_end_fit_override if cfg.t_end_fit_override is not None else t_end_auto
    
    mask_decay = (t >= t_start) & (t <= t_end)
    
    rt60 = 0.0
    t_intersect = t[-1]
    noise_level = -100.0
    p_decay = None
    l_trend = np.full_like(l_ir, np.mean(l_ir))
    
    if np.any(mask_decay):
        p_decay = np.polyfit(t[mask_decay], l_ir[mask_decay], 1)
        
        # Calculate RT60: decay by 60 dB (extrapolated)
        if abs(p_decay[0]) > 1e-3:
            rt60 = 60.0 / abs(p_decay[0])
        
        # Noise floor (last 10%)
        t_noise_start = t[-1] * 0.9
        noise_level = np.mean(l_ir[t >= t_noise_start])
        
        # Intersection point
        if abs(p_decay[0]) > 1e-6:
            t_intersect = (noise_level - p_decay[1]) / p_decay[0]
        
        # Trend line
        l_trend = np.zeros_like(l_ir)
        l_trend[t < t_start] = np.polyval(p_decay, t_start)
        mask_before_intersect = (t >= t_start) & (t <= t_intersect)
        l_trend[mask_before_intersect] = np.polyval(p_decay, t[mask_before_intersect])
        l_trend[t > t_intersect] = noise_level

    return DecayCurveResult(
        t=t,
        l_ir=l_ir,
        l_trend=l_trend,
        rt60_s=rt60,
        t_start_fit=t_start,
        t_end_fit=t_end,
        t_intersect=t_intersect,
        noise_level=noise_level,
        p_decay=p_decay,
        rir_bp=rir_bp,
        sample_rate=fs,
    )


def analyze_flutter(
    impulse_response: np.ndarray,
    config: Optional[FlutterDetectionConfig] = None,
) -> FlutterEchoResult:
    """
    Performs flutter echo detection (pipeline according to DAGA 2023).
    """
    cfg = config or FlutterDetectionConfig()
    fs = cfg.sample_rate
    rir = np.asarray(impulse_response, dtype=float)
    t = np.arange(len(rir)) / fs

    # 1) Bandpass
    b_bp, a_bp = _butter_bandpass(cfg.band_center_hz, cfg.band_q, fs, order=4)
    rir_bp = signal.filtfilt(b_bp, a_bp, rir)

    # 2) Level-time representation (envelope in dB)
    window_len = int(0.002 * fs)
    energy = rir_bp**2
    smoothed_energy = np.convolve(energy, np.ones(window_len)/window_len, mode='same')
    l_ir = 10 * np.log10(smoothed_energy + 1e-12)

    # 3) Trend correction (reverberation isolation)
    rt60 = 0.0
    t_intersect = 0.0
    noise_level = -100.0
    p_decay = None
    if cfg.method == "regression":
        # Regression: Use override values if set
        t_start = cfg.t_start_fit_override if cfg.t_start_fit_override is not None else cfg.t_start_fit
        t_end_auto = min(t[-1] * 0.8, 1.2)
        t_end_fit = cfg.t_end_fit_override if cfg.t_end_fit_override is not None else t_end_auto
        
        # Check if manual slope is set
        if cfg.manual_slope is not None and cfg.manual_intercept is not None:
            # Use manual values
            p_decay = np.array([cfg.manual_slope, cfg.manual_intercept])
            
            # RT60 from manual slope
            if abs(cfg.manual_slope) > 1e-3:
                rt60 = 60.0 / abs(cfg.manual_slope)
            
            # Noise floor: Override or automatic
            if cfg.noise_level_override is not None:
                noise_level = cfg.noise_level_override
            else:
                t_noise_start = t[-1] * 0.9
                noise_level = np.mean(l_ir[t >= t_noise_start])
            
            # Intersection point
            if abs(p_decay[0]) > 1e-6:
                t_intersect = (noise_level - p_decay[1]) / p_decay[0]
            else:
                t_intersect = t[-1]
            
            # Trend line
            l_trend = np.zeros_like(l_ir)
            l_trend[t < t_start] = np.polyval(p_decay, t_start)
            mask_before_intersect = (t >= t_start) & (t <= t_intersect)
            l_trend[mask_before_intersect] = np.polyval(p_decay, t[mask_before_intersect])
            l_trend[t > t_intersect] = noise_level
            
        else:
            # Automatic regression
            mask_decay = (t >= t_start) & (t <= t_end_fit)
            
            if np.any(mask_decay):
                p_decay = np.polyfit(t[mask_decay], l_ir[mask_decay], 1)
                
                # Calculate RT60: decay by 60 dB (extrapolated)
                if abs(p_decay[0]) > 1e-3:
                    rt60 = 60.0 / abs(p_decay[0])
                
                # Noise floor: Override or automatic (last 10%)
                if cfg.noise_level_override is not None:
                    noise_level = cfg.noise_level_override
                else:
                    t_noise_start = t[-1] * 0.9
                    noise_level = np.mean(l_ir[t >= t_noise_start])
                
                # Intersection point
                if abs(p_decay[0]) > 1e-6:
                    t_intersect = (noise_level - p_decay[1]) / p_decay[0]
                else:
                    t_intersect = t[-1]
                
                # Trend line L_dif
                l_trend = np.zeros_like(l_ir)
                l_trend[t < t_start] = np.polyval(p_decay, t_start)
                mask_before_intersect = (t >= t_start) & (t <= t_intersect)
                l_trend[mask_before_intersect] = np.polyval(p_decay, t[mask_before_intersect])
                l_trend[t > t_intersect] = noise_level
            else:
                l_trend = np.full_like(l_ir, np.mean(l_ir))
                t_intersect = t[-1]

        l_fe_db = l_ir - l_trend
        # Time windowing for analysis: Store the actually used values
        t_start_analysis = t_start
        t_end_analysis = t_intersect
        mask_analysis = (t >= t_start_analysis) & (t <= t_end_analysis)
        analysis_signal = l_fe_db[mask_analysis]
        
    else: # Schroeder method
        env_sch = schroeder_integration(energy)
        l_trend = 10 * np.log10(env_sch + 1e-12)
        l_fe_db = l_ir - l_trend
        # Time windowing for analysis: Store the actually used values
        t_start_analysis = cfg.t_start_fit
        t_end_analysis = t[-1]
        mask_analysis = (t >= t_start_analysis)
        analysis_signal = l_fe_db[mask_analysis]
        t_intersect = t[-1]
        # RT60 from Schroeder curve (simplified)
        # Here we skip a complex RT60 estimation for Schroeder
        rt60 = 0.0

    # 4) Autocorrelation (ACF)
    if len(analysis_signal) < 100:
        # Fallback values for time windowing
        t_start_analysis = cfg.t_start_fit_override if cfg.t_start_fit_override is not None else cfg.t_start_fit
        t_end_analysis = t[-1]
        
        return FlutterEchoResult(distances=np.array([]), amplitudes=np.array([]), peaks=[], 
                                main_distance_m=0, distance_uncertainty_m=0, 
                                repetition_frequency_hz=0, relative_amplitude=0, 
                                flutter_tonality_hz=cfg.band_center_hz, severity="n/a", detected=False,
                                rt60_s=rt60, rir_raw=rir, rir_bp=rir_bp, l_fe=l_fe_db, t=t, l_ir=l_ir, l_trend=l_trend,
                                t_start_analysis=t_start_analysis, t_end_analysis=t_end_analysis)

    analysis_signal -= np.mean(analysis_signal)
    acf_full = signal.correlate(analysis_signal, analysis_signal, mode="full")
    acf = acf_full[acf_full.size // 2 :]
    acf /= (acf[0] + 1e-12)
    
    # Optional: Windowing of ACF before FFT to reduce spectral leakage
    # Hann window reduces side lobes and makes peaks sharper
    if len(acf) > 10:
        hann_window = np.hanning(len(acf))
        acf_windowed = acf * hann_window
    else:
        acf_windowed = acf

    # 5) FFT -> Repetition spectrum
    # Zero-padding for better frequency resolution (as in paper)
    # More points = sharper peaks in distogram
    n_fft = len(acf_windowed) * cfg.fft_zero_padding_factor
    # Round to next power of 2 for efficient FFT
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))
    
    spec = np.abs(np.fft.rfft(acf_windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)

    # 6) Frequencies -> Distances
    valid_mask = (freqs > 0) & (freqs < fs / 2)
    freqs_valid = freqs[valid_mask]
    distances_all = SOUND_SPEED / (2.0 * freqs_valid + 1e-10)
    amplitudes_all = spec[valid_mask]

    # Filter range
    dist_mask = (distances_all >= cfg.min_distance_m) & (distances_all <= cfg.max_distance_m)
    distances = distances_all[dist_mask]
    amplitudes = amplitudes_all[dist_mask]

    # Peak detection
    # With zero-padding, peaks are sharper, so we can use stricter criteria
    peaks = []
    if len(amplitudes) > 0:
        height_thr = cfg.peak_rel_height * np.max(amplitudes)
        prominence_thr = cfg.peak_prominence * np.max(amplitudes)
        # Minimum distance between peaks (in samples) for clearer separation
        # With 4x zero-padding we have ~4x more samples, so we can set a minimum distance
        # Minimum distance: ~0.1m at 6m max = ~1.7% of distance axis
        min_distance_samples = max(1, int(len(amplitudes) / 60))  # At least ~1.7% of distance axis
        peak_idx, peak_props = signal.find_peaks(
            amplitudes, 
            height=height_thr, 
            prominence=prominence_thr,
            distance=min_distance_samples
        )
        
        for i, idx in enumerate(peak_idx):
            dist = distances[idx]
            amp = amplitudes[idx]
            peaks.append(PeakInfo(
                distance_m=dist,
                amplitude=amp,
                repetition_frequency_hz=distance_to_frequency(dist),
                is_main=False # will be set after sorting
            ))

        # Sort peaks by distance (ascending)
        peaks.sort(key=lambda p: p.distance_m)
        
        # Mark main peak (the one with highest amplitude)
        if peaks:
            max_amp_peak = max(peaks, key=lambda p: p.amplitude)
            max_amp_peak.is_main = True

    main_peak = max(peaks, key=lambda p: p.amplitude) if peaks else None
    main_distance = main_peak.distance_m if main_peak else 0.0
    main_amp = main_peak.amplitude if main_peak else 0.0

    return FlutterEchoResult(
        distances=distances,
        amplitudes=amplitudes,
        peaks=peaks,
        main_distance_m=main_distance,
        distance_uncertainty_m=0.1,
        repetition_frequency_hz=distance_to_frequency(main_distance),
        relative_amplitude=main_amp,
        flutter_tonality_hz=cfg.band_center_hz,
        severity="audible" if len(peaks) > 0 else "not audible",
        detected=len(peaks) > 0,
        rt60_s=rt60,
        t=t,
        l_ir=l_ir,
        l_trend=l_trend,
        rir_raw=rir,
        rir_bp=rir_bp,
        l_fe=l_fe_db,
        acf=acf,
        t_intersect=t_intersect,
        noise_level=noise_level,
        p_decay=p_decay,
        t_start_analysis=t_start_analysis,
        t_end_analysis=t_end_analysis,
        peak_distances=[p.distance_m for p in peaks],
        peak_amplitudes=[p.amplitude for p in peaks],
    )


# ============================================================
# WRAPPER CLASS FOR GUI
# ============================================================

class FlutterEchoDetector:
    """
    Wrapper for the flutter echo detection algorithm.
    """
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
    
    def analyze(self, impulse_response: np.ndarray) -> FlutterEchoResult:
        """
        Analyze room impulse response for flutter echoes.
        """
        cfg = FlutterDetectionConfig(sample_rate=self.sample_rate)
        return analyze_flutter(impulse_response, cfg)


# ============================================================
# HELPER FUNCTIONS (CONVERSION)
# ============================================================

def distance_to_samples(distance_m: float, sample_rate: int) -> int:
    """Convert distance (m) to samples (round trip)."""
    time_s = (2 * distance_m) / SOUND_SPEED
    return int(time_s * sample_rate)


def samples_to_distance(samples: int, sample_rate: int) -> float:
    """Convert samples to distance (m)."""
    time_s = samples / sample_rate
    return (time_s * SOUND_SPEED) / 2


def frequency_to_distance(freq_hz: float) -> float:
    """Convert repetition frequency to distance."""
    if freq_hz <= 0:
        return 0
    return SOUND_SPEED / (2 * freq_hz)


def distance_to_frequency(distance_m: float) -> float:
    """Convert distance to repetition frequency."""
    if distance_m <= 0:
        return 0
    return SOUND_SPEED / (2 * distance_m)
