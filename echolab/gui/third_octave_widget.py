"""
Third-Octave Analysis Widget

Displays:
- Time-dependent third-octave impulse responses as heatmap
- Individual third-octave time courses
- Export and playback functions

Based on IEC 61260 filterbank.
"""

from typing import Optional, Callable
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFrame, QSpinBox,
    QTabWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal, Qt, QThread, Slot
import pyqtgraph as pg

from ..core.audio_io import AudioFile, save_audio
from ..core.third_octave import ThirdOctaveFilterbank, ThirdOctaveBand, IEC_61260_CENTER_FREQUENCIES
from ..utils.formatting import format_frequency, format_db, format_time


class FilterbankWorker(QThread):
    """
    Worker thread for filterbank computation.
    
    Prevents UI freezing during heavy DSP work.
    """
    
    finished = Signal(list)  # List of ThirdOctaveBand
    progress = Signal(int, int)  # current, total
    error = Signal(str)
    
    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        use_zero_phase: bool = False,
    ):
        super().__init__()
        self.data = data
        self.sample_rate = sample_rate
        self.use_zero_phase = use_zero_phase
        self._cancelled = False
    
    def run(self):
        """Run filterbank computation."""
        try:
            fb = ThirdOctaveFilterbank(
                self.sample_rate,
                use_zero_phase=self.use_zero_phase,
            )
            
            bands = []
            total = len(fb.center_frequencies)
            
            for i, fc in enumerate(fb.center_frequencies):
                if self._cancelled:
                    return
                
                band = fb.filter_single_band(self.data, fc)
                bands.append(band)
                self.progress.emit(i + 1, total)
            
            self.finished.emit(bands)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def cancel(self):
        """Cancel computation."""
        self._cancelled = True


class ThirdOctaveWidget(QWidget):
    """
    Widget for third-octave analysis and impulse response display.
    
    Features:
    - Calculation of third-octave filtering for selected regions
    - Heatmap display of all bands over time
    - Individual display of each third-octave band
    - Export as WAV
    - Playback of individual bands
    
    Signals:
        analysisComplete: Emitted when analysis is complete
        bandSelected: Emitted when a band is selected (frequency)
    """
    
    analysisComplete = Signal()
    bandSelected = Signal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._audio: Optional[AudioFile] = None
        self._bands: list[ThirdOctaveBand] = []
        self._worker: Optional[FilterbankWorker] = None
        self._selected_band_index = 0
        
        # Playback
        self._playback_device = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.Shape.StyledPanel)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        
        # Analysis settings
        toolbar_layout.addWidget(QLabel("Time Resolution:"))
        self.time_res_spin = QSpinBox()
        self.time_res_spin.setRange(1, 100)
        self.time_res_spin.setValue(10)
        self.time_res_spin.setSuffix(" ms")
        toolbar_layout.addWidget(self.time_res_spin)
        
        toolbar_layout.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["Minimum Phase (causal)", "Zero Phase (linear)"])
        self.phase_combo.setToolTip(
            "Minimum Phase: Causal IIR filtering, nonlinear phase\n"
            "Zero Phase: Forward-backward filtering, linear phase, non-causal"
        )
        toolbar_layout.addWidget(self.phase_combo)
        
        toolbar_layout.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        toolbar_layout.addWidget(self.progress_bar)
        
        # Action buttons
        self.btn_analyze = QPushButton("Start Analysis")
        self.btn_analyze.clicked.connect(self._start_analysis)
        self.btn_analyze.setEnabled(False)
        toolbar_layout.addWidget(self.btn_analyze)
        
        layout.addWidget(toolbar)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Band list
        band_frame = QFrame()
        band_layout = QVBoxLayout(band_frame)
        band_layout.setContentsMargins(4, 4, 4, 4)
        
        band_layout.addWidget(QLabel("Third-Octave Bands (IEC 61260):"))
        self.band_list = QListWidget()
        self.band_list.currentRowChanged.connect(self._on_band_selected)
        band_layout.addWidget(self.band_list)
        
        # Export and playback buttons
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("WAV Export")
        self.btn_export.clicked.connect(self._export_band)
        self.btn_export.setEnabled(False)
        btn_layout.addWidget(self.btn_export)
        
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._play_band)
        self.btn_play.setEnabled(False)
        btn_layout.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_playback)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)
        
        band_layout.addLayout(btn_layout)
        
        self.btn_export_all = QPushButton("Export All Bands...")
        self.btn_export_all.clicked.connect(self._export_all_bands)
        self.btn_export_all.setEnabled(False)
        band_layout.addWidget(self.btn_export_all)
        
        splitter.addWidget(band_frame)
        
        # Right: Visualization tabs
        viz_tabs = QTabWidget()
        
        # Tab 1: Heatmap (all bands over time)
        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_widget)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        
        self.heatmap_plot = pg.PlotWidget()
        self.heatmap_plot.setBackground('#1a1a2e')
        self.heatmap_plot.setLabel('left', 'Frequenz', units='Hz')
        self.heatmap_plot.setLabel('bottom', 'Zeit', units='s')
        self.heatmap_img = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_img)
        
        # Colorbar for heatmap
        self.heatmap_colorbar = pg.ColorBarItem(
            values=(-60, 0),
            colorMap=self._create_colormap(),
            label='dB',
        )
        self.heatmap_colorbar.setImageItem(self.heatmap_img)
        
        heatmap_layout.addWidget(self.heatmap_plot)
        
        # dB range controls
        db_layout = QHBoxLayout()
        db_layout.addWidget(QLabel("dB Min:"))
        self.heatmap_db_min = QSpinBox()
        self.heatmap_db_min.setRange(-120, -10)
        self.heatmap_db_min.setValue(-60)
        self.heatmap_db_min.valueChanged.connect(self._update_heatmap_levels)
        db_layout.addWidget(self.heatmap_db_min)
        
        db_layout.addWidget(QLabel("Max:"))
        self.heatmap_db_max = QSpinBox()
        self.heatmap_db_max.setRange(-60, 20)
        self.heatmap_db_max.setValue(0)
        self.heatmap_db_max.valueChanged.connect(self._update_heatmap_levels)
        db_layout.addWidget(self.heatmap_db_max)
        db_layout.addStretch()
        
        heatmap_layout.addLayout(db_layout)
        viz_tabs.addTab(heatmap_widget, "All Bands (Heatmap)")
        
        # Tab 2: Single band view
        single_widget = QWidget()
        single_layout = QVBoxLayout(single_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot for filtered signal
        self.signal_plot = pg.PlotWidget(title="Filtered Signal")
        self.signal_plot.setBackground('#1a1a2e')
        self.signal_plot.setLabel('left', 'Amplitude')
        self.signal_plot.setLabel('bottom', 'Time', units='s')
        self.signal_curve = self.signal_plot.plot(pen=pg.mkPen('#4cc9f0', width=1))
        single_layout.addWidget(self.signal_plot)
        
        # Plot for envelope/impulse response
        self.envelope_plot = pg.PlotWidget(title="Envelope (Impulse Response)")
        self.envelope_plot.setBackground('#1a1a2e')
        self.envelope_plot.setLabel('left', 'Amplitude')
        self.envelope_plot.setLabel('bottom', 'Time', units='s')
        self.envelope_curve = self.envelope_plot.plot(pen=pg.mkPen('#f72585', width=2))
        single_layout.addWidget(self.envelope_plot)
        
        # Level display toggle
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Display:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(["Linear", "dB"])
        self.level_combo.currentTextChanged.connect(self._update_single_band_view)
        level_layout.addWidget(self.level_combo)
        level_layout.addStretch()
        
        self.band_info_label = QLabel("")
        self.band_info_label.setStyleSheet("font-family: monospace;")
        level_layout.addWidget(self.band_info_label)
        
        single_layout.addLayout(level_layout)
        viz_tabs.addTab(single_widget, "Single Band")
        
        # Tab 3: Filter information
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        self.filter_info_text = pg.PlotWidget(title="Filter Frequency Response")
        self.filter_info_text.setBackground('#1a1a2e')
        self.filter_info_text.setLabel('left', 'Magnitude', units='dB')
        self.filter_info_text.setLabel('bottom', 'Frequency', units='Hz')
        self.filter_info_text.setLogMode(x=True, y=False)
        self.filter_mag_curve = self.filter_info_text.plot(pen=pg.mkPen('#4cc9f0', width=2))
        info_layout.addWidget(self.filter_info_text)
        
        self.filter_details_label = QLabel(
            "Select a third-octave band from the list\n"
            "to display the filter characteristics."
        )
        self.filter_details_label.setStyleSheet("font-family: monospace; padding: 8px;")
        info_layout.addWidget(self.filter_details_label)
        
        viz_tabs.addTab(info_widget, "Filter Info")
        
        splitter.addWidget(viz_tabs)
        splitter.setSizes([200, 600])
        
        layout.addWidget(splitter, stretch=1)
    
    def _create_colormap(self) -> pg.ColorMap:
        """Create colormap for heatmap."""
        colors = [
            (0.0, (0, 0, 50)),
            (0.25, (20, 50, 150)),
            (0.5, (100, 100, 200)),
            (0.75, (200, 150, 100)),
            (1.0, (255, 255, 200)),
        ]
        
        positions = [c[0] for c in colors]
        rgb_colors = [c[1] for c in colors]
        
        return pg.ColorMap(positions, rgb_colors)
    
    def set_audio_selection(
        self,
        audio: AudioFile,
        start_sample: int,
        end_sample: int,
        channel: int = 0,
    ):
        """
        Set audio selection for analysis.
        
        Args:
            audio: AudioFile object
            start_sample: Start sample index
            end_sample: End sample index
            channel: Channel to analyze
        """
        self._audio = audio
        self._start_sample = start_sample
        self._end_sample = end_sample
        self._channel = channel
        
        self.btn_analyze.setEnabled(True)
        self._bands = []
        self._update_band_list()
    
    def _start_analysis(self):
        """Start filterbank analysis."""
        if self._audio is None:
            return
        
        # Get selected data
        data = self._audio.get_channel(self._channel)
        data = data[self._start_sample:self._end_sample]
        
        # Check for zero phase option
        use_zero_phase = self.phase_combo.currentIndex() == 1
        
        # Disable UI during computation
        self.btn_analyze.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self._worker = FilterbankWorker(
            data,
            self._audio.sample_rate,
            use_zero_phase,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_complete)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()
    
    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    @Slot(list)
    def _on_analysis_complete(self, bands: list):
        """Handle analysis completion."""
        self._bands = bands
        self.progress_bar.hide()
        self.btn_analyze.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_export_all.setEnabled(True)
        
        self._update_band_list()
        self._update_heatmap()
        
        if self._bands:
            self.band_list.setCurrentRow(0)
        
        self.analysisComplete.emit()
    
    @Slot(str)
    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        self.progress_bar.hide()
        self.btn_analyze.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Analysefehler",
            f"Fehler bei der Terzbandanalyse:\n{error_msg}"
        )
    
    def _update_band_list(self):
        """Update the band list widget."""
        self.band_list.clear()
        
        for band in self._bands:
            rms_db = band.rms_db()
            item_text = f"{format_frequency(band.center_frequency):>10} | {format_db(rms_db)}"
            item = QListWidgetItem(item_text)
            self.band_list.addItem(item)
    
    def _on_band_selected(self, row: int):
        """Handle band selection."""
        if row < 0 or row >= len(self._bands):
            return
        
        self._selected_band_index = row
        band = self._bands[row]
        
        self._update_single_band_view()
        self._update_filter_info(band)
        
        self.bandSelected.emit(band.center_frequency)
    
    def _update_single_band_view(self):
        """Update single band visualization."""
        if not self._bands or self._selected_band_index >= len(self._bands):
            return
        
        band = self._bands[self._selected_band_index]
        
        # Time axis
        time = np.arange(len(band.filtered_signal)) / band.sample_rate
        
        # Filtered signal
        self.signal_curve.setData(time, band.filtered_signal)
        
        # Envelope
        envelope = band.envelope(method="hilbert")
        
        if self.level_combo.currentText() == "dB":
            # Convert to dB
            envelope_db = 20 * np.log10(np.maximum(envelope, 1e-10))
            self.envelope_curve.setData(time, envelope_db)
            self.envelope_plot.setLabel('left', 'Pegel', units='dB')
        else:
            self.envelope_curve.setData(time, envelope)
            self.envelope_plot.setLabel('left', 'Amplitude')
        
        # Update info label
        self.band_info_label.setText(
            f"fc = {format_frequency(band.center_frequency)} | "
            f"RMS = {format_db(band.rms_db())} | "
            f"Peak = {format_db(20 * np.log10(np.max(np.abs(band.filtered_signal)) + 1e-10))}"
        )
    
    def _update_filter_info(self, band: ThirdOctaveBand):
        """Update filter information display."""
        info = band.band_info
        
        # Get filter frequency response
        if self._audio is not None:
            fb = ThirdOctaveFilterbank(
                self._audio.sample_rate,
                use_zero_phase=self.phase_combo.currentIndex() == 1,
            )
            
            try:
                freqs, mag_db, phase = fb.frequency_response(band.center_frequency)
                self.filter_mag_curve.setData(freqs, mag_db)
                self.filter_info_text.setXRange(
                    np.log10(max(20, info.lower_frequency / 2)),
                    np.log10(min(20000, info.upper_frequency * 2)),
                )
                self.filter_info_text.setYRange(-60, 5)
            except Exception:
                pass
        
        # Update details label
        details = (
            f"Mittenfrequenz:     {format_frequency(info.center_frequency)}\n"
            f"Untere Grenze:      {format_frequency(info.lower_frequency)} (-3 dB)\n"
            f"Obere Grenze:       {format_frequency(info.upper_frequency)} (-3 dB)\n"
            f"Bandbreite:         {format_frequency(info.bandwidth)}\n"
            f"Q-Faktor:           {info.quality_factor:.2f}\n"
            f"Filterordnung:      {info.filter_order}\n"
            f"Filtertyp:          {info.filter_type.capitalize()}\n"
            f"Gruppenlaufzeit:    {info.group_delay_at_center:.1f} Samples @ fc\n"
            f"Lineare Phase:      {'Ja' if info.phase_linear else 'Nein'}\n"
            f"\n"
            f"Hinweis: {'Zero-Phase-Filterung aktiv (nicht kausal)' if info.phase_linear else 'Kausale IIR-Filterung (nichtlineare Phase)'}"
        )
        self.filter_details_label.setText(details)
    
    def _update_heatmap(self):
        """Update heatmap with all bands."""
        if not self._bands:
            return
        
        time_res_ms = self.time_res_spin.value()
        
        # Get time-varying levels for all bands
        sample_rate = self._bands[0].sample_rate
        samples_per_frame = int(sample_rate * time_res_ms / 1000)
        num_frames = len(self._bands[0].filtered_signal) // samples_per_frame
        
        if num_frames < 1:
            return
        
        # Build level matrix
        levels = np.zeros((len(self._bands), num_frames))
        
        for i, band in enumerate(self._bands):
            for frame in range(num_frames):
                start = frame * samples_per_frame
                end = start + samples_per_frame
                segment = band.filtered_signal[start:end]
                rms = np.sqrt(np.mean(segment ** 2))
                levels[i, frame] = 20 * np.log10(rms + 1e-10)
        
        # Get dB range
        db_min = self.heatmap_db_min.value()
        db_max = self.heatmap_db_max.value()
        
        # Normalize for display
        levels_normalized = (levels - db_min) / (db_max - db_min)
        levels_normalized = np.clip(levels_normalized, 0, 1)
        
        # Set image data
        self.heatmap_img.setImage(levels_normalized.T)
        
        # Set axes
        time_extent = num_frames * time_res_ms / 1000
        freqs = np.array([b.center_frequency for b in self._bands])
        
        self.heatmap_img.setRect(0, 0, time_extent, len(self._bands))
        
        # Update axis ticks for frequency
        freq_ticks = []
        for i, f in enumerate(freqs):
            if f in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]:
                freq_ticks.append((i, format_frequency(f)))
        
        left_axis = self.heatmap_plot.getAxis('left')
        left_axis.setTicks([freq_ticks])
        
        self.heatmap_colorbar.setLevels((db_min, db_max))
    
    def _update_heatmap_levels(self):
        """Update heatmap dB levels."""
        self._update_heatmap()
    
    def _export_band(self):
        """Export selected band as WAV."""
        if not self._bands or self._selected_band_index >= len(self._bands):
            return
        
        band = self._bands[self._selected_band_index]
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Terzband exportieren",
            f"terzband_{int(band.center_frequency)}Hz.wav",
            "WAV Files (*.wav)",
        )
        
        if filename:
            try:
                save_audio(band.filtered_signal, filename, band.sample_rate)
                QMessageBox.information(
                    self,
                    "Export erfolgreich",
                    f"Terzband {format_frequency(band.center_frequency)} exportiert nach:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export-Fehler", str(e))
    
    def _export_all_bands(self):
        """Export all bands to a directory."""
        if not self._bands:
            return
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Zielverzeichnis für Terzband-Export",
        )
        
        if directory:
            try:
                from pathlib import Path
                
                for band in self._bands:
                    filename = Path(directory) / f"terzband_{int(band.center_frequency)}Hz.wav"
                    save_audio(band.filtered_signal, filename, band.sample_rate)
                
                QMessageBox.information(
                    self,
                    "Export erfolgreich",
                    f"{len(self._bands)} Terzbänder exportiert nach:\n{directory}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export-Fehler", str(e))
    
    def _play_band(self):
        """Play selected band audio."""
        if not self._bands or self._selected_band_index >= len(self._bands):
            return
        
        band = self._bands[self._selected_band_index]
        
        try:
            import sounddevice as sd
            
            # Stop any current playback
            sd.stop()
            
            # Play the filtered signal
            sd.play(band.filtered_signal.astype(np.float32), band.sample_rate)
            
            self.btn_stop.setEnabled(True)
            
        except Exception as e:
            QMessageBox.warning(self, "Wiedergabe-Fehler", str(e))
    
    def _stop_playback(self):
        """Stop audio playback."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        
        self.btn_stop.setEnabled(False)
    
    def get_bands(self) -> list[ThirdOctaveBand]:
        """Get computed bands."""
        return self._bands
    
    def get_selected_band(self) -> Optional[ThirdOctaveBand]:
        """Get currently selected band."""
        if self._bands and 0 <= self._selected_band_index < len(self._bands):
            return self._bands[self._selected_band_index]
        return None

