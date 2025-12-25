"""
Main window of the Echolab application - Simplified version

Structure:
- Tab 1: Time domain (large) + Spectrogram
- Tab 2: FFT analysis of selected region
- Tab 3: Third-octave impulse responses
"""

from typing import Optional
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFileDialog, QMessageBox, QLabel,
    QStatusBar, QApplication, QPushButton, QFrame,
    QComboBox, QSplitter, QListWidget,
    QListWidgetItem, QProgressBar,
)
from PySide6.QtCore import Qt, QSettings, Signal, Slot, QThread
from PySide6.QtGui import QAction, QKeySequence, QWheelEvent, QShortcut
import pyqtgraph as pg

from ..core.audio_io import AudioFile, load_audio, save_audio
from ..core.spectral import compute_spectrogram, SpectrogramConfig
from ..core.third_octave import ThirdOctaveFilterbank, ThirdOctaveBand
from ..utils.formatting import format_time, format_frequency, format_db
from .flutter_echo_widget import FlutterEchoWidget

class ShiftZoomPlotWidget(pg.PlotWidget):
    """
    Custom PlotWidget with Shift key for Y-axis control.
    
    - Default: Only X-axis zoomable and pannable
    - With Shift: Only Y-axis zoomable and pannable
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default: Only X-axis zoomable
        self.setMouseEnabled(x=True, y=False)
        self._shift_active = False
    
    def wheelEvent(self, ev: QWheelEvent):
        """Override wheel event for Shift-Y-zoom."""
        modifiers = ev.modifiers()
        
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Shift pressed: Make Y-axis zoomable
            self.setMouseEnabled(x=False, y=True)
            super().wheelEvent(ev)
            # Back to default
            self.setMouseEnabled(x=True, y=False)
        else:
            # Normal: Only X-axis
            super().wheelEvent(ev)
    
    def mousePressEvent(self, ev):
        """Mouse pressed - check Shift for Y-axis pan."""
        modifiers = ev.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._shift_active = True
            self.setMouseEnabled(x=False, y=True)
        else:
            self._shift_active = False
            self.setMouseEnabled(x=True, y=False)
        super().mousePressEvent(ev)
    
    def mouseMoveEvent(self, ev):
        """Mouse moved - update Shift status."""
        modifiers = ev.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            if not self._shift_active:
                self._shift_active = True
                self.setMouseEnabled(x=False, y=True)
        else:
            if self._shift_active:
                self._shift_active = False
                self.setMouseEnabled(x=True, y=False)
        super().mouseMoveEvent(ev)


class FilterWorker(QThread):
    """Worker for third-octave calculation in background."""
    finished = Signal(list)
    progress = Signal(int, int)
    
    def __init__(self, data, sample_rate):
        super().__init__()
        self.data = data
        self.sample_rate = sample_rate
    
    def run(self):
        fb = ThirdOctaveFilterbank(self.sample_rate)
        bands = []
        total = len(fb.center_frequencies)
        for i, fc in enumerate(fb.center_frequencies):
            band = fb.filter_single_band(self.data, fc)
            bands.append(band)
            self.progress.emit(i + 1, total)
        self.finished.emit(bands)


class LogAxis(pg.AxisItem):
    """Custom axis for standardized acoustic frequency markers."""
    def tickValues(self, minVal, maxVal, size):
        # Define fixed standard frequencies for labeling (Hz)
        freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        # Convert to internal log space (since setLogMode(x=True) is active)
        ticks = [np.log10(f) for f in freqs if minVal <= np.log10(f) <= maxVal]
        return [(1.0, ticks)]

    def tickStrings(self, values, scale, spacing):
        res = []
        for v in values:
            hz = 10**v
            if hz >= 1000:
                res.append(f"{hz/1000:g}k")
            else:
                res.append(f"{hz:g}")
        return res


class MainWindow(QMainWindow):
    """Simplified main window."""
    
    def __init__(self):
        super().__init__()
        
        self._audio: Optional[AudioFile] = None
        self._selection_start = 0
        self._selection_end = 0
        self._bands: list[ThirdOctaveBand] = []
        self._worker: Optional[FilterWorker] = None
        self._selection_stream = None  # For playback of selected region
        self._band_stream = None  # For playback of third-octave band
        
        self._init_ui()
        self._apply_theme()
    
    def _next_tab(self):
        """Switch to next tab."""
        self.tabs.setCurrentIndex((self.tabs.currentIndex() + 1) % self.tabs.count())

    def _prev_tab(self):
        """Switch to previous tab."""
        self.tabs.setCurrentIndex((self.tabs.currentIndex() - 1) % self.tabs.count())

    def _init_ui(self):
        """Build UI."""
        self.setWindowTitle("Echolab")
        self.setMinimumSize(1000, 700)
        
        # Shortcuts for tab navigation (Global)
        self.shortcut_tab_next = QShortcut(QKeySequence("Tab"), self)
        self.shortcut_tab_next.activated.connect(self._next_tab)
        
        self.shortcut_tab_prev = QShortcut(QKeySequence("Shift+Tab"), self)
        self.shortcut_tab_prev.activated.connect(self._prev_tab)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Toolbar at top
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 8)
        
        btn_open = QPushButton("Import")
        btn_open.clicked.connect(self._open_file)
        toolbar_layout.addWidget(btn_open)
        toolbar_layout.addStretch()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(self.file_label)
        
        layout.addWidget(toolbar)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)
        
        # === Tab 1: Time domain + Spectrogram ===
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(0, 0, 0, 0)
        tab1_layout.setSpacing(0)
        
        # === CONTROL BAR AT TOP ===
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(8, 8, 8, 4)
        control_layout.setSpacing(8)
        
        control_layout.addStretch()
        
        # Buttons at top right
        self.btn_play_selection = QPushButton("Play")
        self.btn_play_selection.clicked.connect(self._play_selection)
        self.btn_play_selection.setEnabled(False)
        control_layout.addWidget(self.btn_play_selection)
        
        self.btn_stop_selection = QPushButton("Stop")
        self.btn_stop_selection.clicked.connect(self._stop_selection_playback)
        self.btn_stop_selection.setEnabled(False)
        control_layout.addWidget(self.btn_stop_selection)
        
        tab1_layout.addWidget(control_bar)
        
        # === SPLITTER FOR PLOTS ===
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(1)
        
        # === WAVEFORM PLOT ===
        waveform_container = QWidget()
        waveform_layout = QVBoxLayout(waveform_container)
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        waveform_layout.setSpacing(0)
        
        self.waveform_plot = ShiftZoomPlotWidget()
        self.waveform_plot.setBackground('#1e1e2e')
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.getAxis('bottom').setStyle(showValues=False)
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.setXRange(0, 1)
        self.waveform_plot.getAxis('left').setWidth(60)
        self.waveform_plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.waveform_plot.getPlotItem().setMenuEnabled(False)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen('#89b4fa', width=1))
        
        # Selection region
        self.selection_region = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(137, 180, 250, 50),
            pen=pg.mkPen('#89b4fa', width=2),
        )
        self.selection_region.sigRegionChanged.connect(self._on_selection_changed)
        self.waveform_plot.addItem(self.selection_region)
        
        waveform_layout.addWidget(self.waveform_plot)
        splitter.addWidget(waveform_container)
        
        # === SPECTROGRAM PLOT ===
        spectro_container = QWidget()
        spectro_layout = QVBoxLayout(spectro_container)
        spectro_layout.setContentsMargins(0, 0, 0, 0)
        spectro_layout.setSpacing(0)
        
        self.spectro_plot = ShiftZoomPlotWidget()
        self.spectro_plot.setBackground('#1e1e2e')
        self.spectro_plot.setLabel('left', 'Frequency', units='Hz')
        self.spectro_plot.setLabel('bottom', 'Time', units='s')
        self.spectro_plot.setXRange(0, 1)
        self.spectro_plot.setYRange(0, 20000)
        self.spectro_plot.getAxis('left').setWidth(60)
        self.spectro_plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.spectro_plot.getPlotItem().setMenuEnabled(False)
        
        self.spectro_img = pg.ImageItem()
        self.spectro_plot.addItem(self.spectro_img)
        self._create_acoustic_colormap()
        
        spectro_layout.addWidget(self.spectro_plot)
        splitter.addWidget(spectro_container)
        splitter.setSizes([500, 500])
        
        tab1_layout.addWidget(splitter)
        
        # === INFO AT BOTTOM LEFT ===
        info_bar = QWidget()
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(8, 4, 8, 8)
        
        self.selection_label = QLabel("Selection: - ")
        self.selection_label.setStyleSheet("padding: 6px 10px; font-family: monospace; font-size: 11px; background: #181825; border-radius: 4px; color: #cdd6f4;")
        info_layout.addWidget(self.selection_label)
        info_layout.addStretch()
        
        tab1_layout.addWidget(info_bar)
        self.tabs.addTab(tab1, "Time Domain")
        
        # === Tab 2: FFT Analysis ===
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(0, 0, 0, 0)
        tab2_layout.setSpacing(0)
        
        # === CONTROL BAR AT TOP ===
        fft_control_bar = QWidget()
        fft_control_layout = QHBoxLayout(fft_control_bar)
        fft_control_layout.setContentsMargins(8, 8, 8, 4)
        fft_control_layout.setSpacing(8)
        
        fft_control_layout.addStretch()
        
        # FFT Controls at top right (same position as Play buttons in Tab 1)
        fft_control_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["1024", "2048", "4096", "8192", "16384"])
        self.fft_size_combo.setCurrentText("4096")
        self.fft_size_combo.currentTextChanged.connect(self._update_fft)
        fft_control_layout.addWidget(self.fft_size_combo)
        
        fft_control_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.window_combo.currentTextChanged.connect(self._update_fft)
        fft_control_layout.addWidget(self.window_combo)
        
        tab2_layout.addWidget(fft_control_bar)
        
        # === FFT PLOT ===
        fft_plot_container = QWidget()
        fft_plot_layout = QVBoxLayout(fft_plot_container)
        fft_plot_layout.setContentsMargins(0, 0, 0, 0)
        fft_plot_layout.setSpacing(0)
        
        # FFT PLOT with custom log axis
        self.fft_plot = pg.PlotWidget(axisItems={'bottom': LogAxis(orientation='bottom')})
        self.fft_plot.setBackground('#1e1e2e')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.setLabel('left', 'Magnitude', units='dB')
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.getAxis('left').setWidth(60)  # Gleiche Breite wie Tab 1
        self.fft_plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.fft_plot.getPlotItem().setMenuEnabled(False)
        self.fft_plot.setXRange(np.log10(20), np.log10(20000), padding=0)
        self.fft_plot.setYRange(-100, 0)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen('#89b4fa', width=1.5))
        
        fft_plot_layout.addWidget(self.fft_plot)
        tab2_layout.addWidget(fft_plot_container, stretch=1)
        
        # === INFO AT BOTTOM LEFT (same position and styling as Tab 1) ===
        fft_info_bar = QWidget()
        fft_info_layout = QHBoxLayout(fft_info_bar)
        fft_info_layout.setContentsMargins(8, 4, 8, 8)
        
        self.fft_info = QLabel("Select a region in the Time Domain tab")
        self.fft_info.setStyleSheet("padding: 6px 10px; font-family: monospace; font-size: 11px; background: #181825; border-radius: 4px; color: #cdd6f4;")
        fft_info_layout.addWidget(self.fft_info)
        fft_info_layout.addStretch()
        
        tab2_layout.addWidget(fft_info_bar)
        
        self.tabs.addTab(tab2, "FFT Analysis")
        
        # === Tab 3: Third-Octave Impulse Responses ===
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(8, 8, 8, 8)
        
        # Controls
        terz_controls = QHBoxLayout()
        
        self.btn_analyze = QPushButton("Start Third-Octave Analysis")
        self.btn_analyze.clicked.connect(self._start_analysis)
        self.btn_analyze.setEnabled(False)
        terz_controls.addWidget(self.btn_analyze)
        
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        terz_controls.addWidget(self.progress)
        
        terz_controls.addStretch()
        
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._play_band)
        self.btn_play.setEnabled(False)
        terz_controls.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_playback)
        self.btn_stop.setEnabled(False)
        terz_controls.addWidget(self.btn_stop)
        
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._export_band)
        self.btn_export.setEnabled(False)
        terz_controls.addWidget(self.btn_export)
        
        tab3_layout.addLayout(terz_controls)
        
        # Splitter for list and plot
        terz_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Band list
        self.band_list = QListWidget()
        self.band_list.setMaximumWidth(180)
        self.band_list.currentRowChanged.connect(self._on_band_selected)
        terz_splitter.addWidget(self.band_list)
        
        # Impulse response plot
        self.impulse_plot = pg.PlotWidget(title="Third-Octave Impulse Response")
        self.impulse_plot.setBackground('#1e1e2e')
        self.impulse_plot.showGrid(x=True, y=True, alpha=0.3)
        self.impulse_plot.setLabel('left', 'Amplitude')
        self.impulse_plot.setLabel('bottom', 'Time', units='s')
        # Disable right-click menu
        self.impulse_plot.getPlotItem().setMenuEnabled(False)
        self.impulse_curve = self.impulse_plot.plot(pen=pg.mkPen('#f38ba8', width=1))
        self.envelope_curve = self.impulse_plot.plot(pen=pg.mkPen('#fab387', width=2))
        
        terz_splitter.addWidget(self.impulse_plot)
        
        terz_splitter.setSizes([150, 600])
        tab3_layout.addWidget(terz_splitter, stretch=1)
        
        self.tabs.addTab(tab3, "Third-Octave Impulse Responses")
        
        # === Tab 4: Room-Analysis ===
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.setContentsMargins(8, 8, 8, 8)
        
        # Controls
        flutter_controls = QHBoxLayout()
        
        self.btn_flutter_analyze = QPushButton("Room-Analysis")
        self.btn_flutter_analyze.clicked.connect(self._start_flutter_analysis)
        self.btn_flutter_analyze.setEnabled(False)
        flutter_controls.addWidget(self.btn_flutter_analyze)
        
        flutter_controls.addStretch()
        
        # Info label
        self.flutter_info_label = QLabel("Load an impulse response and select a region")
        self.flutter_info_label.setStyleSheet("color: #888;")
        flutter_controls.addWidget(self.flutter_info_label)
        
        tab4_layout.addLayout(flutter_controls)
        
        # Flutter Echo Widget (now Room Analysis)
        self.flutter_widget = FlutterEchoWidget()
        tab4_layout.addWidget(self.flutter_widget, stretch=1)
        
        self.tabs.addTab(tab4, "Room Analysis")

        
        # Statusbar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("")
        self.statusBar.addWidget(self.status_label)
        
        # Tab change handler
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # View synchronization: Both plots always synchronized
        self._syncing = False  # Prevents infinite loop
        self.waveform_plot.sigXRangeChanged.connect(self._sync_from_waveform)
        self.spectro_plot.sigXRangeChanged.connect(self._sync_from_spectro)
        
        # Drag & Drop
        self.setAcceptDrops(True)
    
    def _apply_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:disabled {
                background-color: #181825;
                color: #585b70;
            }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                background-color: #1e1e2e;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #a6adc8;
                padding: 10px 20px;
                border: 1px solid #45475a;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e2e;
                color: #89b4fa;
                border-bottom: 2px solid #89b4fa;
            }
            QListWidget {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 6px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QProgressBar {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
            }
            QSplitter::handle {
                background-color: #45475a;
            }
            QStatusBar {
                background-color: #181825;
            }
        """)
    
    def _open_file(self):
        """Open file dialog."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio (*.wav *.mp3);;All Files (*)"
        )
        if filename:
            self._load_file(filename)
    
    def _load_file(self, filepath: str):
        """Load audio file."""
        try:
            self.status_label.setText(f"Loading {Path(filepath).name}...")
            QApplication.processEvents()
            
            self._audio = load_audio(filepath)
            
            # Display waveform
            data = self._audio.get_channel(0)  # Left channel / Mono
            time = np.arange(len(data)) / self._audio.sample_rate
            
            # Downsampling for display
            if len(data) > 50000:
                factor = len(data) // 50000
                data_ds = data[::factor]
                time_ds = time[::factor]
                self.waveform_curve.setData(time_ds, data_ds)
            else:
                self.waveform_curve.setData(time, data)
            
            # Set axes
            self.waveform_plot.setXRange(0, self._audio.duration_seconds)
            self.waveform_plot.setYRange(-1, 1)
            
            # Set selection to full range + bounds
            self.selection_region.setBounds([0, self._audio.duration_seconds])
            self.selection_region.setRegion([0, self._audio.duration_seconds])
            
            # Calculate spectrogram
            self._update_spectrogram()
            
            # Update FFT if FFT tab is active
            if self.tabs.currentIndex() == 1:  # FFT Tab
                self._update_fft()
            
            # Update UI
            self.file_label.setText(
                f"{self._audio.file_path.name} | "
                f"{self._audio.sample_rate} Hz | "
                f"{'Stereo' if self._audio.channels == 2 else 'Mono'} | "
                f"{format_time(self._audio.duration_seconds)}"
            )
            self.btn_analyze.setEnabled(True)
            self.btn_play_selection.setEnabled(True)
            self.btn_flutter_analyze.setEnabled(True)
            self.status_label.setText("")
            self._bands = []
            self.band_list.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")
    
    def _create_acoustic_colormap(self):
        """Create classic spectrogram colormap (jet-like) as in pyqtgraph-spectrographer."""
        # Try to use built-in "jet" colormap from pyqtgraph first
        try:
            cmap = pg.colormap.get('jet')
            self.spectro_img.setColorMap(cmap)
        except:
            # Fallback: Custom "jet"-like colormap
            # Classic "jet" colormap for spectrograms: blue -> cyan -> green -> yellow -> red
            # Optimized for acoustic analyses with good frequency resolution
            colors = [
                (0.0, (0, 0, 128)),         # Dark blue (very quiet)
                (0.25, (0, 0, 255)),        # Blue (quiet)
                (0.5, (0, 255, 255)),       # Cyan (medium)
                (0.75, (255, 255, 0)),      # Yellow (loud)
                (1.0, (255, 0, 0)),         # Red (very loud)
            ]
            
            positions = [c[0] for c in colors]
            rgb_colors = [c[1] for c in colors]
            
            cmap = pg.ColorMap(positions, rgb_colors)
            self.spectro_img.setColorMap(cmap)
    
    def _update_spectrogram(self):
        """Calculate and display spectrogram."""
        if self._audio is None:
            return
        
        data = self._audio.get_channel(0)
        # Very high resolution for acoustic analyses: 16384 FFT, 95% overlap
        config = SpectrogramConfig(fft_size=16384, overlap_percent=95.0)
        result = compute_spectrogram(data, self._audio.sample_rate, config)
        
        # Convert to dB with extended range for better dynamics
        db_min = -100  # Extended range for better visibility
        db_max = 0
        magnitude_db = result.magnitude_db(min_db=db_min)
        
        # Convert to uint8 for pyqtgraph (0-255) with extended dB range
        img_data = (magnitude_db - db_min) / (db_max - db_min)  # 0-1
        img_data = np.clip(img_data, 0, 1)
        img_uint8 = (img_data * 255).astype(np.uint8)
        
        # Set image (transposed: time on X, frequency on Y)
        self.spectro_img.setImage(img_uint8.T)
        
        # Scale to real time/frequency axes
        time_max = result.times[-1] if len(result.times) > 0 else 1
        freq_max = result.frequencies[-1] if len(result.frequencies) > 0 else 1
        
        # Transform for correct axis scaling
        tr = pg.QtGui.QTransform()
        tr.scale(time_max / img_uint8.shape[1], freq_max / img_uint8.shape[0])
        self.spectro_img.setTransform(tr)
        
        # Set levels for better display
        self.spectro_img.setLevels([0, 255])
        
        self.spectro_plot.setXRange(0, self._audio.duration_seconds)
        self.spectro_plot.setYRange(0, min(self._audio.sample_rate / 2, 20000))
    
    def _on_selection_changed(self):
        """Selection changed."""
        if self._audio is None:
            return
        
        region = self.selection_region.getRegion()
        self._selection_start = max(0, region[0])
        self._selection_end = min(self._audio.duration_seconds, region[1])
        
        duration = self._selection_end - self._selection_start
        start_samples = int(self._selection_start * self._audio.sample_rate)
        end_samples = int(self._selection_end * self._audio.sample_rate)
        
        self.selection_label.setText(
            f"Selection: {format_time(self._selection_start)} – "
            f"{format_time(self._selection_end)} "
            f"({format_time(duration)} | {end_samples - start_samples:,} Samples)"
        )
        
        # Enable play button if valid selection
        if self._audio is not None and duration > 0.01:
            self.btn_play_selection.setEnabled(True)
        else:
            self.btn_play_selection.setEnabled(False)
        
        # Update FFT if tab is active
        if self.tabs.currentIndex() == 1:
            self._update_fft()
    
    def _on_tab_changed(self, index: int):
        """Tab changed."""
        if index == 1:  # FFT Tab
            self._update_fft()
    
    def _sync_from_waveform(self):
        """Synchronize waveform zoom → spectrogram."""
        if self._syncing or self._audio is None:
            return
        self._syncing = True
        view_range = self.waveform_plot.viewRange()
        self.spectro_plot.setXRange(view_range[0][0], view_range[0][1], padding=0)
        self._syncing = False
    
    def _sync_from_spectro(self):
        """Synchronize spectrogram zoom → waveform."""
        if self._syncing or self._audio is None:
            return
        self._syncing = True
        view_range = self.spectro_plot.viewRange()
        self.waveform_plot.setXRange(view_range[0][0], view_range[0][1], padding=0)
        self._syncing = False
    
    def _update_fft(self):
        """Calculate FFT for selected region."""
        if self._audio is None:
            return
        
        # Get selected region
        start_sample = int(self._selection_start * self._audio.sample_rate)
        end_sample = int(self._selection_end * self._audio.sample_rate)
        data = self._audio.get_channel(0)[start_sample:end_sample]
        
        if len(data) < 64:
            self.fft_info.setText("Selection too short for FFT")
            return
        
        # FFT parameters
        fft_size = int(self.fft_size_combo.currentText())
        window_name = self.window_combo.currentText()
        
        # Window
        if window_name == "hann":
            window = np.hanning(len(data))
        elif window_name == "hamming":
            window = np.hamming(len(data))
        elif window_name == "blackman":
            window = np.blackman(len(data))
        else:
            window = np.ones(len(data))
        
        # FFT
        windowed = data * window
        spectrum = np.fft.rfft(windowed, n=fft_size)
        magnitude = np.abs(spectrum)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Frequency axis
        freqs = np.fft.rfftfreq(fft_size, 1 / self._audio.sample_rate)
        
        # Plot (X-axis is log-scaled)
        self.fft_curve.setData(freqs[1:], magnitude_db[1:])  # Without DC
        f_max = min(self._audio.sample_rate / 2, 20000)
        self.fft_plot.setXRange(np.log10(20), np.log10(f_max), padding=0)
        
        # Auto-scale Y-axis based on actual data
        valid_magnitude = magnitude_db[1:]  # Exclude DC
        if len(valid_magnitude) > 0:
            y_min = np.min(valid_magnitude)
            y_max = np.max(valid_magnitude)
            # Add padding: 10% above max, and ensure minimum is at least 20 dB below max
            y_padding = max((y_max - y_min) * 0.1, 5.0)
            y_max_scaled = y_max + y_padding
            y_min_scaled = min(y_min - y_padding, y_max - 80)  # At least 80 dB range, but not above max
            self.fft_plot.setYRange(y_min_scaled, y_max_scaled)
        else:
            # Fallback to default range
            self.fft_plot.setYRange(-100, 0)
        
        # Info
        self.fft_info.setText(
            f"Samples: {len(data):,} | "
            f"Frequency resolution: {self._audio.sample_rate / fft_size:.2f} Hz"
        )
    
    def _start_analysis(self):
        """Start third-octave analysis."""
        if self._audio is None:
            return
        
        # Get selected region
        start_sample = int(self._selection_start * self._audio.sample_rate)
        end_sample = int(self._selection_end * self._audio.sample_rate)
        data = self._audio.get_channel(0)[start_sample:end_sample]
        
        if len(data) < 1000:
            QMessageBox.warning(self, "Note", "Selection too short for third-octave analysis")
            return
        
        # UI
        self.btn_analyze.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.status_label.setText("Calculating third-octave bands...")
        
        # Start worker
        self._worker = FilterWorker(data, self._audio.sample_rate)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.start()
    
    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        """Update progress."""
        self.progress.setMaximum(total)
        self.progress.setValue(current)
    
    @Slot(list)
    def _on_analysis_done(self, bands: list):
        """Analysis complete."""
        self._bands = bands
        self.btn_analyze.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.progress.hide()
        self.status_label.setText(f"Analysis complete - {len(bands)} bands")
        
        # Fill list
        self.band_list.clear()
        for band in bands:
            text = f"{format_frequency(band.center_frequency)}"
            self.band_list.addItem(text)
        
        if bands:
            self.band_list.setCurrentRow(0)
    
    def _on_band_selected(self, row: int):
        """Third-octave band selected."""
        if row < 0 or row >= len(self._bands):
            return
        
        band = self._bands[row]
        
        # Time
        time = np.arange(len(band.filtered_signal)) / band.sample_rate
        
        # Plot signal
        self.impulse_curve.setData(time, band.filtered_signal)
        
        # Envelope
        envelope = band.envelope(method="hilbert")
        self.envelope_curve.setData(time, envelope)
        
        # Title
        self.impulse_plot.setTitle(
            f"Third-Octave {format_frequency(band.center_frequency)} | "
            f"RMS: {format_db(band.rms_db())}"
        )
    
    def _apply_fade(self, data: np.ndarray, sample_rate: int, fade_ms: float = 10.0) -> np.ndarray:
        """Apply fade-in and fade-out to avoid clicks."""
        if len(data) == 0:
            return data
        
        # Number of samples for fade
        fade_samples = int(sample_rate * fade_ms / 1000.0)
        fade_samples = min(fade_samples, len(data) // 2)  # Maximum half the length
        
        if fade_samples < 1:
            return data
        
        # Create copy
        faded = data.copy()
        
        # Fade-In (first fade_samples)
        if len(faded.shape) == 1:
            # Mono
            fade_in = np.linspace(0, 1, fade_samples)
            faded[:fade_samples] *= fade_in
        else:
            # Stereo/Multi-Channel
            fade_in = np.linspace(0, 1, fade_samples)
            faded[:fade_samples, :] *= fade_in[:, np.newaxis]
        
        # Fade-Out (last fade_samples)
        if len(faded.shape) == 1:
            # Mono
            fade_out = np.linspace(1, 0, fade_samples)
            faded[-fade_samples:] *= fade_out
        else:
            # Stereo/Multi-Channel
            fade_out = np.linspace(1, 0, fade_samples)
            faded[-fade_samples:, :] *= fade_out[:, np.newaxis]
        
        return faded
    
    def _play_band(self):
        """Play selected band."""
        row = self.band_list.currentRow()
        if row < 0 or row >= len(self._bands):
            return
        
        band = self._bands[row]
        
        # Stop previous playback
        self._stop_playback()
        
        try:
            import sounddevice as sd
            
            # Stop if something is already playing
            sd.stop()
            
            # Apply fade-in/out to avoid clicks
            faded_data = self._apply_fade(band.filtered_signal, band.sample_rate, fade_ms=10.0)
            
            # Play
            self._band_stream = sd.play(faded_data.astype(np.float32), band.sample_rate)
            
            self.btn_stop.setEnabled(True)
            self.status_label.setText(f"Playing {format_frequency(band.center_frequency)}...")
            
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", str(e))
            self._stop_playback()
    
    def _stop_playback(self):
        """Stop playback."""
        try:
            import sounddevice as sd
            sd.stop()
            self._band_stream = None
        except:
            pass
        
        self.btn_stop.setEnabled(False)
        self.status_label.setText("")
    
    def _play_selection(self):
        """Play selected region."""
        if self._audio is None:
            return
        
        # Stop previous playback
        self._stop_selection_playback()
        
        try:
            import sounddevice as sd
            
            # Get selected region
            start_sample = int(self._selection_start * self._audio.sample_rate)
            end_sample = int(self._selection_end * self._audio.sample_rate)
            
            if end_sample <= start_sample:
                return
            
            # Extract data
            if self._audio.channels == 1:
                data = self._audio.data[start_sample:end_sample]
            else:
                # Stereo: both channels
                data = self._audio.data[start_sample:end_sample]
            
            # Apply fade-in/out to avoid clicks
            faded_data = self._apply_fade(data, self._audio.sample_rate, fade_ms=10.0)
            
            # Play
            self._selection_stream = sd.play(faded_data.astype(np.float32), self._audio.sample_rate)
            
            # Update UI
            self.btn_play_selection.setEnabled(False)
            self.btn_stop_selection.setEnabled(True)
            duration = self._selection_end - self._selection_start
            self.status_label.setText(f"Playing selected region ({format_time(duration)})")
            
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", str(e))
            self._stop_selection_playback()
    
    def _stop_selection_playback(self):
        """Stop playback of selected region."""
        try:
            import sounddevice as sd
            sd.stop()
            self._selection_stream = None
        except:
            pass
        
        # Update UI
        self.btn_play_selection.setEnabled(True)
        self.btn_stop_selection.setEnabled(False)
        self.status_label.setText("")
    
    def _export_band(self):
        """Export selected band."""
        row = self.band_list.currentRow()
        if row < 0 or row >= len(self._bands):
            return
        
        band = self._bands[row]
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Third-Octave Band",
            f"third_octave_{int(band.center_frequency)}Hz.wav",
            "WAV (*.wav)"
        )
        
        if filename:
            try:
                save_audio(band.filtered_signal, filename, band.sample_rate)
                QMessageBox.information(self, "Export", f"Exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def dragEnterEvent(self, event):
        """Drag & Drop."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.wav', '.mp3')):
                    event.acceptProposedAction()
                    return
    
    def dropEvent(self, event):
        """File dropped."""
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            if filepath.lower().endswith(('.wav', '.mp3')):
                self._load_file(filepath)
                break
    

    def _start_flutter_analysis(self):
        """Start Room Analysis."""
        if self._audio is None:
            return
        
        # Get selected region
        start_sample = int(self._selection_start * self._audio.sample_rate)
        end_sample = int(self._selection_end * self._audio.sample_rate)
        data = self._audio.get_channel(0)[start_sample:end_sample]
        
        if len(data) < 1000:
            QMessageBox.warning(self, "Note", "Selection too short for Room Analysis")
            return
        
        # Update UI
        self.btn_flutter_analyze.setEnabled(False)
        self.flutter_info_label.setText("Analysis running...")
        self.status_label.setText("Calculating Room Analysis...")
        QApplication.processEvents()
        
        try:
            # Pass data to widget and analyze
            self.flutter_widget.set_audio_data(data, self._audio.sample_rate)
            self.flutter_widget.analyze()
            
            # Result info
            if self.flutter_widget._result and self.flutter_widget._result.detected:
                self.flutter_info_label.setText(
                    f"Resonances/Echos detected: {self.flutter_widget._result.main_distance_m:.2f} m | "
                    f"{self.flutter_widget._result.severity}"
                )
            else:
                self.flutter_info_label.setText("No significant resonances detected")
            
            self.status_label.setText("Room Analysis complete")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Analysis failed: {e}")
            self.flutter_info_label.setText("Analysis failed")
        
        finally:
            self.btn_flutter_analyze.setEnabled(True)
    
    def closeEvent(self, event):
        """On close."""
        # Stop playback
        self._stop_playback()
        self._stop_selection_playback()
        
        # Terminate worker
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        event.accept()
