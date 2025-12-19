"""
Spektrogramm-Widget

Zeigt Zeit-Frequenz-Darstellung mit:
- Logarithmischer Frequenzachse
- Konfigurierbaren FFT-Parametern
- Synchronisation mit Waveform-Widget

Verwendet pyqtgraph ImageItem für effiziente Darstellung.
"""

from typing import Optional
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QFrame, QSlider, QGroupBox,
)
from PySide6.QtCore import Signal, Qt
import pyqtgraph as pg

from ..core.audio_io import AudioFile
from ..core.spectral import compute_spectrogram, SpectrogramConfig, frequency_to_log_scale
from ..utils.formatting import format_frequency


class SpectrogramWidget(QWidget):
    """
    Widget zur Darstellung von Spektrogrammen.
    
    Features:
    - Logarithmische oder lineare Frequenzachse
    - Konfigurierbare FFT-Parameter (Größe, Fenster, Überlappung)
    - Farbskala-Anpassung (dB-Bereich)
    - Synchronisation mit Waveform-View
    
    Signals:
        parameterChanged: Emittiert wenn FFT-Parameter geändert werden
    """
    
    parameterChanged = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._audio: Optional[AudioFile] = None
        self._current_channel = 0
        self._spectrogram_data: Optional[np.ndarray] = None
        
        # Default config
        self._config = SpectrogramConfig(
            fft_size=2048,
            window="hann",
            overlap_percent=75.0,
        )
        
        self._db_min = -90.0
        self._db_max = 0.0
        self._use_log_freq = True
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Settings panel
        settings_frame = QFrame()
        settings_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        settings_layout = QHBoxLayout(settings_frame)
        settings_layout.setContentsMargins(8, 4, 8, 4)
        
        # FFT Size
        settings_layout.addWidget(QLabel("FFT:"))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["512", "1024", "2048", "4096", "8192", "16384"])
        self.fft_combo.setCurrentText("2048")
        self.fft_combo.currentTextChanged.connect(self._on_fft_changed)
        settings_layout.addWidget(self.fft_combo)
        
        # Window function
        settings_layout.addWidget(QLabel("Fenster:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "kaiser", "rectangular"])
        self.window_combo.setCurrentText("hann")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        settings_layout.addWidget(self.window_combo)
        
        # Overlap
        settings_layout.addWidget(QLabel("Überlappung:"))
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItems(["25%", "50%", "75%", "87.5%"])
        self.overlap_combo.setCurrentText("75%")
        self.overlap_combo.currentTextChanged.connect(self._on_overlap_changed)
        settings_layout.addWidget(self.overlap_combo)
        
        settings_layout.addStretch()
        
        # Frequency scale toggle
        settings_layout.addWidget(QLabel("Frequenz:"))
        self.freq_scale_combo = QComboBox()
        self.freq_scale_combo.addItems(["Logarithmisch", "Linear"])
        self.freq_scale_combo.currentTextChanged.connect(self._on_freq_scale_changed)
        settings_layout.addWidget(self.freq_scale_combo)
        
        # dB range
        settings_layout.addWidget(QLabel("dB Min:"))
        self.db_min_spin = QSpinBox()
        self.db_min_spin.setRange(-120, -10)
        self.db_min_spin.setValue(-90)
        self.db_min_spin.valueChanged.connect(self._on_db_range_changed)
        settings_layout.addWidget(self.db_min_spin)
        
        settings_layout.addWidget(QLabel("Max:"))
        self.db_max_spin = QSpinBox()
        self.db_max_spin.setRange(-60, 20)
        self.db_max_spin.setValue(0)
        self.db_max_spin.valueChanged.connect(self._on_db_range_changed)
        settings_layout.addWidget(self.db_max_spin)
        
        layout.addWidget(settings_frame)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a2e')
        self.plot_widget.setLabel('left', 'Frequenz', units='Hz')
        self.plot_widget.setLabel('bottom', 'Zeit', units='s')
        
        # Image item for spectrogram
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        
        # Color bar
        self.colorbar = pg.ColorBarItem(
            values=(-90, 0),
            colorMap=self._create_colormap(),
            label='dB',
        )
        self.colorbar.setImageItem(self.img_item)
        
        # Enable mouse interaction (X only for sync)
        self.plot_widget.setMouseEnabled(x=True, y=False)
        
        # Selection region (synced with waveform)
        self.selection_region: Optional[pg.LinearRegionItem] = None
        
        layout.addWidget(self.plot_widget, stretch=1)
        
        # Info label
        self.info_label = QLabel("FFT: 2048 | Fenster: hann | Auflösung: -- Hz / -- ms")
        self.info_label.setStyleSheet("color: #888; padding: 4px; font-family: monospace;")
        layout.addWidget(self.info_label)
    
    def _create_colormap(self) -> pg.ColorMap:
        """Create colormap for spectrogram."""
        # Use a perceptually uniform colormap similar to viridis
        colors = [
            (0.0, (13, 8, 135)),      # Dark blue
            (0.25, (84, 2, 163)),     # Purple
            (0.5, (139, 90, 43)),     # Brown/Orange
            (0.75, (253, 174, 97)),   # Yellow/Orange
            (1.0, (252, 253, 191)),   # Light yellow
        ]
        
        positions = [c[0] for c in colors]
        rgb_colors = [c[1] for c in colors]
        
        return pg.ColorMap(positions, rgb_colors)
    
    def set_audio(self, audio: AudioFile, channel: int = 0):
        """
        Set audio data for spectrogram calculation.
        
        Args:
            audio: AudioFile object
            channel: Channel index to display
        """
        self._audio = audio
        self._current_channel = channel
        self._compute_and_display()
    
    def _compute_and_display(self):
        """Compute spectrogram and update display."""
        if self._audio is None:
            return
        
        # Get channel data
        data = self._audio.get_channel(self._current_channel)
        
        # Compute spectrogram
        result = compute_spectrogram(data, self._audio.sample_rate, self._config)
        
        # Convert to dB
        magnitude_db = result.magnitude_db(min_db=self._db_min)
        
        # Apply log frequency scale if selected
        if self._use_log_freq:
            magnitude_db, frequencies = frequency_to_log_scale(
                magnitude_db,
                result.frequencies,
                num_bins=512,
                f_min=20.0,
                f_max=min(20000, self._audio.sample_rate / 2),
            )
        else:
            frequencies = result.frequencies
        
        self._spectrogram_data = magnitude_db
        
        # Update image
        # Transpose for correct orientation (time on x, frequency on y)
        img_data = magnitude_db.T
        
        # Normalize to 0-1 range for colormap
        img_normalized = (img_data - self._db_min) / (self._db_max - self._db_min)
        img_normalized = np.clip(img_normalized, 0, 1)
        
        self.img_item.setImage(img_normalized.T)
        
        # Set correct transform for axes
        time_range = result.times[-1] - result.times[0] if len(result.times) > 1 else 1
        freq_range = frequencies[-1] - frequencies[0] if len(frequencies) > 1 else 1
        
        # Calculate scale factors
        x_scale = time_range / magnitude_db.shape[1] if magnitude_db.shape[1] > 0 else 1
        y_scale = freq_range / magnitude_db.shape[0] if magnitude_db.shape[0] > 0 else 1
        
        self.img_item.setRect(
            result.times[0] if len(result.times) > 0 else 0,
            frequencies[0] if len(frequencies) > 0 else 0,
            time_range,
            freq_range,
        )
        
        # Update colorbar
        self.colorbar.setLevels((self._db_min, self._db_max))
        
        # Set Y axis (frequency) range
        if self._use_log_freq:
            self.plot_widget.setLogMode(y=True)
            self.plot_widget.setYRange(np.log10(20), np.log10(frequencies[-1]), padding=0)
        else:
            self.plot_widget.setLogMode(y=False)
            self.plot_widget.setYRange(0, frequencies[-1], padding=0)
        
        # Update info label
        freq_res = self._config.frequency_resolution(self._audio.sample_rate)
        time_res = self._config.time_resolution(self._audio.sample_rate) * 1000
        self.info_label.setText(
            f"FFT: {self._config.fft_size} | "
            f"Fenster: {self._config.window} | "
            f"Auflösung: {freq_res:.1f} Hz / {time_res:.1f} ms"
        )
    
    def _on_fft_changed(self, value: str):
        """Handle FFT size change."""
        self._config.fft_size = int(value)
        self._config.hop_size = None  # Recalculate from overlap
        self._config.__post_init__()
        self._compute_and_display()
        self.parameterChanged.emit()
    
    def _on_window_changed(self, value: str):
        """Handle window function change."""
        self._config.window = value
        self._compute_and_display()
        self.parameterChanged.emit()
    
    def _on_overlap_changed(self, value: str):
        """Handle overlap change."""
        overlap_map = {"25%": 25.0, "50%": 50.0, "75%": 75.0, "87.5%": 87.5}
        self._config.overlap_percent = overlap_map.get(value, 75.0)
        self._config.hop_size = None
        self._config.__post_init__()
        self._compute_and_display()
        self.parameterChanged.emit()
    
    def _on_freq_scale_changed(self, value: str):
        """Handle frequency scale change."""
        self._use_log_freq = (value == "Logarithmisch")
        self._compute_and_display()
    
    def _on_db_range_changed(self):
        """Handle dB range change."""
        self._db_min = self.db_min_spin.value()
        self._db_max = self.db_max_spin.value()
        self._compute_and_display()
    
    def set_view_range(self, start_time: float, end_time: float):
        """
        Set X-axis view range (for synchronization with waveform).
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        self.plot_widget.setXRange(start_time, end_time, padding=0)
    
    def set_channel(self, channel: int):
        """
        Set channel to display.
        
        Args:
            channel: Channel index (0 or 1)
        """
        if self._audio is not None and channel != self._current_channel:
            self._current_channel = channel
            self._compute_and_display()
    
    def show_selection(self, start_time: float, end_time: float):
        """
        Show selection region overlay.
        
        Args:
            start_time: Selection start in seconds
            end_time: Selection end in seconds
        """
        if self.selection_region is not None:
            self.plot_widget.removeItem(self.selection_region)
        
        self.selection_region = pg.LinearRegionItem(
            values=[start_time, end_time],
            orientation='vertical',
            brush=pg.mkBrush(100, 150, 200, 50),
            movable=False,
        )
        self.plot_widget.addItem(self.selection_region)
    
    def clear_selection(self):
        """Clear selection overlay."""
        if self.selection_region is not None:
            self.plot_widget.removeItem(self.selection_region)
            self.selection_region = None
    
    @property
    def config(self) -> SpectrogramConfig:
        """Current spectrogram configuration."""
        return self._config

