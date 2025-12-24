"""
Waveform Widget

Displays audio signal in time domain with:
- Zoom and pan
- Precise region marking
- Time and sample display

Uses pyqtgraph for performant display of large datasets.
"""

from typing import Optional, Callable
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QFrame,
)
from PySide6.QtCore import Signal, Qt
import pyqtgraph as pg

from ..core.audio_io import AudioFile
from ..utils.formatting import format_time, samples_to_time_str


class SelectionRegion(pg.LinearRegionItem):
    """
    Extended region for time selection.
    
    Emits signals on change and displays time/sample information.
    """
    
    selectionChanged = Signal(float, float)  # start_time, end_time
    
    def __init__(self, sample_rate: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.sigRegionChanged.connect(self._on_region_changed)
        
        # Styling
        self.setBrush(pg.mkBrush(100, 150, 200, 80))
        self.setHoverBrush(pg.mkBrush(100, 150, 200, 120))
        self.setMovable(True)
        
    def _on_region_changed(self):
        """Emit selection changed signal."""
        region = self.getRegion()
        self.selectionChanged.emit(region[0], region[1])
    
    def get_sample_range(self) -> tuple[int, int]:
        """Get selection as sample indices."""
        region = self.getRegion()
        start_sample = int(region[0] * self.sample_rate)
        end_sample = int(region[1] * self.sample_rate)
        return start_sample, end_sample
    
    def set_sample_range(self, start_sample: int, end_sample: int):
        """Set selection from sample indices."""
        start_time = start_sample / self.sample_rate
        end_time = end_sample / self.sample_rate
        self.setRegion([start_time, end_time])


class WaveformWidget(QWidget):
    """
    Widget for displaying audio waveforms.
    
    Features:
    - Zoomable time axis (mouse wheel + drag)
    - Channel selection for stereo
    - Region marking with time and sample display
    - Fast display through downsampling at large zoom-out
    
    Signals:
        selectionChanged: (start_seconds, end_seconds, start_samples, end_samples)
        viewChanged: (start_time, end_time) - For synchronization with spectrogram
    """
    
    selectionChanged = Signal(float, float, int, int)
    viewChanged = Signal(float, float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._audio: Optional[AudioFile] = None
        self._current_channel = 0
        self._selection: Optional[SelectionRegion] = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.Shape.StyledPanel)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        
        # Channel selector
        toolbar_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Left", "Right"])
        self.channel_combo.setEnabled(False)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        toolbar_layout.addWidget(self.channel_combo)
        
        toolbar_layout.addStretch()
        
        # Selection info
        self.selection_label = QLabel("No selection")
        self.selection_label.setStyleSheet("font-family: monospace;")
        toolbar_layout.addWidget(self.selection_label)
        
        toolbar_layout.addStretch()
        
        # Zoom controls
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.clicked.connect(self._zoom_to_fit)
        toolbar_layout.addWidget(self.btn_zoom_fit)
        
        self.btn_zoom_selection = QPushButton("Zoom Selection")
        self.btn_zoom_selection.clicked.connect(self._zoom_to_selection)
        self.btn_zoom_selection.setEnabled(False)
        toolbar_layout.addWidget(self.btn_zoom_selection)
        
        self.btn_clear_selection = QPushButton("Clear Selection")
        self.btn_clear_selection.clicked.connect(self._clear_selection)
        self.btn_clear_selection.setEnabled(False)
        toolbar_layout.addWidget(self.btn_clear_selection)
        
        layout.addWidget(toolbar)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a2e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.getViewBox().sigRangeChanged.connect(self._on_view_changed)
        
        # Waveform curve
        self.waveform_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#4cc9f0', width=1),
            name='Waveform'
        )
        
        # Crosshair for precise positioning
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#ffffff', width=1, style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#ffffff', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)
        self.vline.hide()
        self.hline.hide()
        
        # Mouse tracking for crosshair
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        # Context menu for selection
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        
        layout.addWidget(self.plot_widget, stretch=1)
        
        # Status bar
        self.status_label = QLabel("Keine Datei geladen")
        self.status_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self.status_label)
    
    def set_audio(self, audio: AudioFile):
        """
        Set audio data for display.
        
        Args:
            audio: AudioFile object with loaded data
        """
        self._audio = audio
        self._current_channel = 0
        
        # Update channel selector
        if audio.channels == 1:
            self.channel_combo.setEnabled(False)
            self.channel_combo.setCurrentIndex(0)
            self.channel_combo.clear()
            self.channel_combo.addItem("Mono")
        else:
            self.channel_combo.setEnabled(True)
            self.channel_combo.clear()
            self.channel_combo.addItems(["Links (L)", "Rechts (R)"])
        
        # Clear any existing selection
        self._clear_selection()
        
        # Update display
        self._update_waveform()
        self._zoom_to_fit()
        
        # Update status
        self.status_label.setText(
            f"{audio.file_path.name} | "
            f"{audio.sample_rate} Hz | "
            f"{'Mono' if audio.channels == 1 else 'Stereo'} | "
            f"{format_time(audio.duration_seconds)}"
        )
    
    def _update_waveform(self):
        """Update waveform display with current channel."""
        if self._audio is None:
            return
        
        # Get channel data
        data = self._audio.get_channel(self._current_channel)
        
        # Create time axis
        time_axis = np.arange(len(data)) / self._audio.sample_rate
        
        # Downsample for display if needed (performance optimization)
        if len(data) > 100000:
            factor = len(data) // 100000
            # Use min/max envelope for accurate peak display
            data_reshaped = data[:len(data) // factor * factor].reshape(-1, factor)
            mins = data_reshaped.min(axis=1)
            maxs = data_reshaped.max(axis=1)
            # Interleave min and max for envelope
            display_data = np.empty(len(mins) * 2)
            display_data[0::2] = mins
            display_data[1::2] = maxs
            time_axis = np.linspace(0, self._audio.duration_seconds, len(display_data))
            self.waveform_curve.setData(time_axis, display_data)
        else:
            self.waveform_curve.setData(time_axis, data)
    
    def _on_channel_changed(self, index: int):
        """Handle channel selection change."""
        self._current_channel = index
        self._update_waveform()
    
    def _on_view_changed(self):
        """Handle view range change for synchronization."""
        if self._audio is None:
            return
        
        view_range = self.plot_widget.getViewBox().viewRange()
        self.viewChanged.emit(view_range[0][0], view_range[0][1])
    
    def _on_mouse_moved(self, pos):
        """Update crosshair position."""
        if self._audio is None:
            return
        
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            self.vline.setPos(mouse_point.x())
            self.hline.setPos(mouse_point.y())
            self.vline.show()
            self.hline.show()
        else:
            self.vline.hide()
            self.hline.hide()
    
    def _on_mouse_clicked(self, event):
        """Handle mouse click for selection creation."""
        if self._audio is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton and event.double():
            # Double-click creates selection at click position
            pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
                time = mouse_point.x()
                
                # Create selection of 0.5 seconds centered at click
                duration = 0.5
                start = max(0, time - duration/2)
                end = min(self._audio.duration_seconds, time + duration/2)
                
                self._create_selection(start, end)
    
    def _create_selection(self, start_time: float, end_time: float):
        """Create or update selection region."""
        if self._audio is None:
            return
        
        # Remove existing selection
        if self._selection is not None:
            self.plot_widget.removeItem(self._selection)
        
        # Create new selection
        self._selection = SelectionRegion(
            self._audio.sample_rate,
            values=[start_time, end_time],
            orientation='vertical',
        )
        self._selection.selectionChanged.connect(self._on_selection_changed)
        self.plot_widget.addItem(self._selection)
        
        # Enable buttons
        self.btn_zoom_selection.setEnabled(True)
        self.btn_clear_selection.setEnabled(True)
        
        # Update display
        self._on_selection_changed(start_time, end_time)
    
    def _on_selection_changed(self, start_time: float, end_time: float):
        """Handle selection region change."""
        if self._audio is None or self._selection is None:
            return
        
        # Clamp to valid range
        start_time = max(0, start_time)
        end_time = min(self._audio.duration_seconds, end_time)
        
        # Calculate sample indices
        start_sample = int(start_time * self._audio.sample_rate)
        end_sample = int(end_time * self._audio.sample_rate)
        
        # Update label
        duration = end_time - start_time
        num_samples = end_sample - start_sample
        self.selection_label.setText(
            f"Selektion: {format_time(start_time)} - {format_time(end_time)} "
            f"({format_time(duration)} | {num_samples:,} Samples)"
        )
        
        # Emit signal
        self.selectionChanged.emit(start_time, end_time, start_sample, end_sample)
    
    def _clear_selection(self):
        """Clear current selection."""
        if self._selection is not None:
            self.plot_widget.removeItem(self._selection)
            self._selection = None
        
        self.selection_label.setText("Keine Selektion")
        self.btn_zoom_selection.setEnabled(False)
        self.btn_clear_selection.setEnabled(False)
    
    def _zoom_to_fit(self):
        """Zoom to show entire waveform."""
        if self._audio is None:
            return
        
        self.plot_widget.setXRange(0, self._audio.duration_seconds, padding=0.02)
        self.plot_widget.setYRange(-1.1, 1.1, padding=0)
    
    def _zoom_to_selection(self):
        """Zoom to show current selection."""
        if self._selection is None:
            return
        
        region = self._selection.getRegion()
        padding = (region[1] - region[0]) * 0.1
        self.plot_widget.setXRange(region[0] - padding, region[1] + padding, padding=0)
    
    def get_selection(self) -> Optional[tuple[float, float, int, int]]:
        """
        Get current selection.
        
        Returns:
            Tuple of (start_time, end_time, start_sample, end_sample) or None
        """
        if self._selection is None or self._audio is None:
            return None
        
        region = self._selection.getRegion()
        start_sample = int(region[0] * self._audio.sample_rate)
        end_sample = int(region[1] * self._audio.sample_rate)
        return (region[0], region[1], start_sample, end_sample)
    
    def get_selected_data(self) -> Optional[np.ndarray]:
        """
        Get audio data for current selection.
        
        Returns:
            Numpy array with selected samples or None
        """
        if self._selection is None or self._audio is None:
            return None
        
        start_sample, end_sample = self._selection.get_sample_range()
        return self._audio.get_time_range(start_sample, end_sample)
    
    def set_view_range(self, start_time: float, end_time: float):
        """
        Set view range (for external synchronization).
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        self.plot_widget.setXRange(start_time, end_time, padding=0)
    
    def create_selection_at(self, start_time: float, end_time: float):
        """
        Create selection programmatically.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        self._create_selection(start_time, end_time)
    
    @property
    def current_channel(self) -> int:
        """Current channel index (0=left/mono, 1=right)."""
        return self._current_channel

