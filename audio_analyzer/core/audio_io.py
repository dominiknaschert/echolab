"""
Audio I/O Module

Loads and saves audio files (WAV, MP3) without implicit signal manipulation.
The user consciously decides on downmix and resampling.

Technical assumptions:
- WAV files are loaded with soundfile (high precision, no conversion)
- MP3 files are decoded with librosa/audioread
- All audio data is returned as float64 numpy arrays (range -1.0 to 1.0)
- Channel order for stereo: [left, right] as (samples, 2) array
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import soundfile as sf


@dataclass
class AudioFile:
    """
    Represents a loaded audio file with all metadata.
    
    The audio signal is NOT automatically modified.
    Downmix/Resampling only occurs on explicit user instruction.
    
    Attributes:
        data: Audio data as numpy array, Shape: (samples,) or (samples, channels)
        sample_rate: Original sample rate of the file
        channels: Number of channels (1=Mono, 2=Stereo)
        duration_seconds: Duration in seconds
        num_samples: Number of samples per channel
        file_path: Path to source file
        format_info: Format information (Subtype, Endianness)
        bit_depth: Bit depth of original (if known)
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
        """Validate data integrity."""
        if self.data.ndim == 1:
            assert self.channels == 1, "1D array must be mono"
        elif self.data.ndim == 2:
            assert self.data.shape[1] == self.channels, "Channel count mismatch"
        else:
            raise ValueError("Audio array must be 1D or 2D")
    
    def get_channel(self, channel: int) -> np.ndarray:
        """
        Extract a single channel.
        
        Args:
            channel: 0 for left/Mono, 1 for right
            
        Returns:
            1D numpy array with the channel samples
        """
        if self.channels == 1:
            if channel != 0:
                raise ValueError("Mono file only has channel 0")
            return self.data if self.data.ndim == 1 else self.data[:, 0]
        else:
            if channel not in (0, 1):
                raise ValueError("Stereo only supports channels 0 and 1")
            return self.data[:, channel]
    
    def time_to_sample(self, time_seconds: float) -> int:
        """Convert time in seconds to sample index."""
        sample = int(time_seconds * self.sample_rate)
        return max(0, min(sample, self.num_samples - 1))
    
    def sample_to_time(self, sample: int) -> float:
        """Convert sample index to time in seconds."""
        return sample / self.sample_rate
    
    def get_time_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        """
        Extract a time range (non-destructive).
        
        Returns:
            Copy of audio data in the specified range
        """
        start = max(0, start_sample)
        end = min(end_sample, self.num_samples)
        
        if self.data.ndim == 1:
            return self.data[start:end].copy()
        else:
            return self.data[start:end, :].copy()


def load_audio(file_path: str | Path) -> AudioFile:
    """
    Load an audio file without implicit conversion.
    
    Supported formats:
    - WAV (all common subtypes: PCM_16, PCM_24, PCM_32, FLOAT)
    - MP3 (via librosa/audioread)
    
    Audio data is returned as float64 in range [-1.0, 1.0].
    NO automatic conversion of sample rate or channel count.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        AudioFile object with all metadata
        
    Raises:
        FileNotFoundError: File does not exist
        ValueError: Unsupported format
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".wav":
        return _load_wav(path)
    elif suffix == ".mp3":
        return _load_mp3(path)
    else:
        raise ValueError(f"Nicht unterstütztes Format: {suffix}")


def _load_wav(path: Path) -> AudioFile:
    """
    Load WAV file with soundfile.
    
    soundfile uses libsndfile and provides precise results
    without unwanted conversions.
    """
    # Read audio and sample rate
    data, sample_rate = sf.read(path, dtype='float64', always_2d=False)
    
    # Read extended metadata
    info = sf.info(path)
    
    # Determine bit depth from subtype
    bit_depth = _extract_bit_depth(info.subtype)
    
    # Determine channel count
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
    Load MP3 file with pydub.
    
    Requires ffmpeg in system PATH for decoding.
    If ffmpeg is not available, an error is raised.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is not installed. Please install with 'pip install pydub'.")
    
    try:
        audio = AudioSegment.from_mp3(path)
    except Exception as e:
        raise RuntimeError(
            f"MP3 could not be loaded: {e}\n"
            "Please ensure ffmpeg is installed:\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    
    # Extract metadata
    sample_rate = audio.frame_rate
    channels = audio.channels
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize to float64 in range [-1, 1]
    # pydub returns int16 or int32
    if audio.sample_width == 2:  # 16-bit
        samples = samples.astype(np.float64) / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples = samples.astype(np.float64) / 2147483648.0
    else:  # 8-bit
        samples = (samples.astype(np.float64) - 128) / 128.0
    
    # Reshape for stereo
    if channels == 2:
        samples = samples.reshape(-1, 2)
        num_samples = samples.shape[0]
    else:
        num_samples = len(samples)
    
    format_info = {
        "format": "MP3",
        "subtype": "MPEG Layer 3",
        "note": "MP3 is lossy, original data cannot be reconstructed",
    }
    
    return AudioFile(
        data=samples,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=num_samples / sample_rate,
        num_samples=num_samples,
        file_path=path,
        format_info=format_info,
        bit_depth=None,  # MP3 has no fixed bit depth
    )


def save_audio(
    data: np.ndarray,
    file_path: str | Path,
    sample_rate: int,
    subtype: Literal["PCM_16", "PCM_24", "PCM_32", "FLOAT"] = "PCM_24",
) -> None:
    """
    Save audio data as WAV file.
    
    Args:
        data: Audio data as numpy array (float64, range -1.0 to 1.0)
        file_path: Target path
        sample_rate: Sample rate
        subtype: WAV subtype for quantization
        
    Raises:
        ValueError: Invalid data or parameters
    """
    path = Path(file_path)
    
    # Validation
    if data.ndim > 2:
        raise ValueError("Audio must be 1D or 2D")
    
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Audio data must be float")
    
    # Clipping warning
    if np.any(np.abs(data) > 1.0):
        import warnings
        warnings.warn(
            "Audio data exceeds [-1.0, 1.0]. Clipping will be applied.",
            UserWarning
        )
        data = np.clip(data, -1.0, 1.0)
    
    sf.write(path, data, sample_rate, subtype=subtype)


def _extract_bit_depth(subtype: str) -> Optional[int]:
    """Extract bit depth from soundfile subtype string."""
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

