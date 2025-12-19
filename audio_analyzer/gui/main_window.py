"""
Hauptfenster der Audio Analyzer Anwendung - Vereinfachte Version

Struktur:
- Tab 1: Zeitbereich (groß) + Spektrogramm
- Tab 2: FFT-Analyse des selektierten Bereichs
- Tab 3: Terzband-Impulsantworten
"""

from typing import Optional
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFileDialog, QMessageBox, QLabel,
    QStatusBar, QApplication, QPushButton, QFrame,
    QComboBox, QSplitter, QSpinBox, QListWidget,
    QListWidgetItem, QProgressBar,
)
from PySide6.QtCore import Qt, QSettings, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
import pyqtgraph as pg

from ..core.audio_io import AudioFile, load_audio, save_audio
from ..core.spectral import compute_spectrogram, SpectrogramConfig
from ..core.third_octave import ThirdOctaveFilterbank, ThirdOctaveBand
from ..utils.formatting import format_time, format_frequency, format_db


class FilterWorker(QThread):
    """Worker für Terzband-Berechnung im Hintergrund."""
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


class MainWindow(QMainWindow):
    """Vereinfachtes Hauptfenster."""
    
    def __init__(self):
        super().__init__()
        
        self._audio: Optional[AudioFile] = None
        self._selection_start = 0
        self._selection_end = 0
        self._bands: list[ThirdOctaveBand] = []
        self._worker: Optional[FilterWorker] = None
        self._selection_stream = None  # Für Wiedergabe des selektierten Bereichs
        self._selection_playback_start_time = 0.0  # Startzeit der Wiedergabe
        self._selection_timer: Optional[QTimer] = None  # Timer für Cursor-Update
        self._is_playing_selection = False  # Flag für aktive Wiedergabe
        self._band_stream = None  # Für Wiedergabe des Terzbands
        self._band_playback_start_time = 0.0  # Startzeit der Terzband-Wiedergabe
        self._band_timer: Optional[QTimer] = None  # Timer für Terzband-Cursor-Update
        self._is_playing_band = False  # Flag für aktive Terzband-Wiedergabe
        self._current_band_duration = 0.0  # Dauer des aktuellen Terzbands
        
        self._init_ui()
        self._apply_theme()
    
    def _init_ui(self):
        """UI aufbauen."""
        self.setWindowTitle("Audio Analyzer")
        self.setMinimumSize(1000, 700)
        
        # Zentrales Widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Toolbar oben
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 8)
        
        btn_open = QPushButton("Import")
        btn_open.clicked.connect(self._open_file)
        toolbar_layout.addWidget(btn_open)
        
        toolbar_layout.addStretch()
        
        self.file_label = QLabel("Keine Datei geladen")
        self.file_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(self.file_label)
        
        layout.addWidget(toolbar)
        
        # Tab Widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)
        
        # === Tab 1: Zeitbereich + Spektrogramm ===
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter für Waveform und Spektrogramm
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Waveform Plot (groß)
        waveform_container = QWidget()
        waveform_layout = QVBoxLayout(waveform_container)
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        
        # Buttons oben rechts
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(8, 8, 8, 8)
        button_layout.addStretch()
        
        self.btn_play_selection = QPushButton("▶ Abspielen")
        self.btn_play_selection.clicked.connect(self._play_selection)
        self.btn_play_selection.setEnabled(False)
        button_layout.addWidget(self.btn_play_selection)
        
        self.btn_stop_selection = QPushButton("■ Stop")
        self.btn_stop_selection.clicked.connect(self._stop_selection_playback)
        self.btn_stop_selection.setEnabled(False)
        button_layout.addWidget(self.btn_stop_selection)
        
        waveform_layout.addWidget(button_container)
        
        self.waveform_plot = pg.PlotWidget(title="Zeitbereich")
        self.waveform_plot.setBackground('#1e1e2e')
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Zeit', units='s')
        # NUR X-Achse zoombar, Y-Achse gelockt
        self.waveform_plot.setMouseEnabled(x=True, y=False)
        self.waveform_plot.setYRange(-1, 1)
        # Feste Y-Achsenbreite für Ausrichtung mit Spektrogramm
        self.waveform_plot.getAxis('left').setWidth(60)
        # Rechtsklick-Menü deaktivieren
        self.waveform_plot.getPlotItem().setMenuEnabled(False)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen('#89b4fa', width=1))
        
        # Playback-Cursor für Wiedergabe
        self.playback_cursor = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen('#f38ba8', width=2),
            movable=False
        )
        self.playback_cursor.setVisible(False)  # Initial versteckt
        self.waveform_plot.addItem(self.playback_cursor)
        
        # Selektion-Region
        self.selection_region = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(137, 180, 250, 50),
            pen=pg.mkPen('#89b4fa', width=2),
        )
        self.selection_region.sigRegionChanged.connect(self._on_selection_changed)
        self.waveform_plot.addItem(self.selection_region)
        
        waveform_layout.addWidget(self.waveform_plot)
        
        # Info-Zeile (nur Selektion)
        selection_info_layout = QHBoxLayout()
        self.selection_label = QLabel("Selektion: - ")
        self.selection_label.setStyleSheet("padding: 4px 8px; font-family: monospace; background: #181825; border-radius: 4px;")
        selection_info_layout.addWidget(self.selection_label)
        selection_info_layout.addStretch()
        waveform_layout.addLayout(selection_info_layout)
        
        splitter.addWidget(waveform_container)
        
        # Spektrogramm Plot
        spectro_container = QWidget()
        spectro_layout = QVBoxLayout(spectro_container)
        spectro_layout.setContentsMargins(0, 0, 0, 0)
        
        self.spectro_plot = pg.PlotWidget(title="Spektrogramm")
        self.spectro_plot.setBackground('#1e1e2e')
        self.spectro_plot.setLabel('left', 'Frequenz', units='Hz')
        self.spectro_plot.setLabel('bottom', 'Zeit', units='s')
        # NUR X-Achse zoombar (synchron mit Waveform), Y-Achse gelockt
        self.spectro_plot.setMouseEnabled(x=True, y=False)
        # Feste Y-Achsenbreite für Ausrichtung mit Waveform
        self.spectro_plot.getAxis('left').setWidth(60)
        # Rechtsklick-Menü deaktivieren
        self.spectro_plot.getPlotItem().setMenuEnabled(False)
        
        self.spectro_img = pg.ImageItem()
        self.spectro_plot.addItem(self.spectro_img)
        
        # Akustik-spezifische Colormap (blau-grün-gelb-rot für bessere Sichtbarkeit)
        self._create_acoustic_colormap()
        
        # Playback-Cursor für Wiedergabe (synchron mit Waveform)
        self.spectro_cursor = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen('#f38ba8', width=2),
            movable=False
        )
        self.spectro_cursor.setVisible(False)  # Initial versteckt
        self.spectro_plot.addItem(self.spectro_cursor)
        
        spectro_layout.addWidget(self.spectro_plot)
        
        splitter.addWidget(spectro_container)
        splitter.setSizes([350, 250])
        
        tab1_layout.addWidget(splitter)
        self.tabs.addTab(tab1, "Zeitbereich & Spektrogramm")
        
        # === Tab 2: FFT Analyse ===
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(8, 8, 8, 8)
        
        # FFT Controls
        fft_controls = QHBoxLayout()
        fft_controls.addWidget(QLabel("FFT-Größe:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["1024", "2048", "4096", "8192", "16384"])
        self.fft_size_combo.setCurrentText("4096")
        self.fft_size_combo.currentTextChanged.connect(self._update_fft)
        fft_controls.addWidget(self.fft_size_combo)
        
        fft_controls.addWidget(QLabel("Fenster:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.window_combo.currentTextChanged.connect(self._update_fft)
        fft_controls.addWidget(self.window_combo)
        
        fft_controls.addStretch()
        tab2_layout.addLayout(fft_controls)
        
        # FFT Plot
        self.fft_plot = pg.PlotWidget(title="FFT - Magnitude Spektrum (selektierter Bereich)")
        self.fft_plot.setBackground('#1e1e2e')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.setLabel('left', 'Magnitude', units='dB')
        self.fft_plot.setLabel('bottom', 'Frequenz', units='Hz')
        self.fft_plot.setLogMode(x=True, y=False)
        # Rechtsklick-Menü deaktivieren
        self.fft_plot.getPlotItem().setMenuEnabled(False)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen('#a6e3a1', width=1.5))
        
        # Professionelle Frequenzmarkierungen für X-Achse
        self._setup_fft_axis_ticks()
        
        tab2_layout.addWidget(self.fft_plot, stretch=1)
        
        self.fft_info = QLabel("Wählen Sie einen Bereich im Zeitbereich-Tab")
        self.fft_info.setStyleSheet("padding: 8px; font-family: monospace;")
        tab2_layout.addWidget(self.fft_info)
        
        self.tabs.addTab(tab2, "FFT Analyse")
        
        # === Tab 3: Terzband-Impulsantworten ===
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(8, 8, 8, 8)
        
        # Controls
        terz_controls = QHBoxLayout()
        
        self.btn_analyze = QPushButton("▶ Terzbandanalyse starten")
        self.btn_analyze.clicked.connect(self._start_analysis)
        self.btn_analyze.setEnabled(False)
        terz_controls.addWidget(self.btn_analyze)
        
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        terz_controls.addWidget(self.progress)
        
        terz_controls.addStretch()
        
        self.btn_play = QPushButton("▶ Abspielen")
        self.btn_play.clicked.connect(self._play_band)
        self.btn_play.setEnabled(False)
        terz_controls.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.clicked.connect(self._stop_playback)
        self.btn_stop.setEnabled(False)
        terz_controls.addWidget(self.btn_stop)
        
        self.btn_export = QPushButton("💾 Exportieren")
        self.btn_export.clicked.connect(self._export_band)
        self.btn_export.setEnabled(False)
        terz_controls.addWidget(self.btn_export)
        
        tab3_layout.addLayout(terz_controls)
        
        # Splitter für Liste und Plot
        terz_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Bandliste
        self.band_list = QListWidget()
        self.band_list.setMaximumWidth(180)
        self.band_list.currentRowChanged.connect(self._on_band_selected)
        terz_splitter.addWidget(self.band_list)
        
        # Impulsantwort Plot
        self.impulse_plot = pg.PlotWidget(title="Terzband-Impulsantwort")
        self.impulse_plot.setBackground('#1e1e2e')
        self.impulse_plot.showGrid(x=True, y=True, alpha=0.3)
        self.impulse_plot.setLabel('left', 'Amplitude')
        self.impulse_plot.setLabel('bottom', 'Zeit', units='s')
        # Rechtsklick-Menü deaktivieren
        self.impulse_plot.getPlotItem().setMenuEnabled(False)
        self.impulse_curve = self.impulse_plot.plot(pen=pg.mkPen('#f38ba8', width=1))
        self.envelope_curve = self.impulse_plot.plot(pen=pg.mkPen('#fab387', width=2))
        
        # Playback-Cursor für Terzband-Wiedergabe
        self.band_cursor = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen('#f38ba8', width=2),
            movable=False
        )
        self.band_cursor.setVisible(False)  # Initial versteckt
        self.impulse_plot.addItem(self.band_cursor)
        terz_splitter.addWidget(self.impulse_plot)
        
        terz_splitter.setSizes([150, 600])
        tab3_layout.addWidget(terz_splitter, stretch=1)
        
        self.tabs.addTab(tab3, "Terzband-Impulsantworten")
        
        # Statusbar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("")
        self.statusBar.addWidget(self.status_label)
        
        # Tab-Wechsel Handler
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # View-Synchronisation: Beide Plots immer synchron
        self._syncing = False  # Verhindert Endlosschleife
        self.waveform_plot.sigXRangeChanged.connect(self._sync_from_waveform)
        self.spectro_plot.sigXRangeChanged.connect(self._sync_from_spectro)
        
        # Drag & Drop
        self.setAcceptDrops(True)
    
    def _apply_theme(self):
        """Dark theme."""
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
    
    def _setup_fft_axis_ticks(self):
        """Initialisiere professionelle Frequenzmarkierungen für FFT-Plot."""
        # Standard-Frequenzen für professionelle akustische Plots (bis 16 kHz)
        self._standard_freqs = [20, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
    def _format_fft_frequency(self, hz: float) -> str:
        """Formatiere Frequenz für FFT-Achse: '1 kHz' statt '1.0 kHz'."""
        if hz >= 1000:
            # Keine Dezimalstellen für kHz
            return f"{int(hz/1000)} kHz"
        else:
            return f"{int(hz)} Hz"
    
    def _update_fft_axis_ticks(self, f_max: float):
        """Aktualisiere Frequenzmarkierungen basierend auf dem Frequenzbereich."""
        # Begrenze auf 16 kHz
        f_max = min(f_max, 16000)
        
        # Wähle relevante Frequenzen im aktuellen Bereich
        relevant_freqs = [f for f in self._standard_freqs if 20 <= f <= f_max]
        
        # Erstelle Ticks mit logarithmischen Positionen
        ticks = []
        for freq in relevant_freqs:
            log_pos = np.log10(freq)
            ticks.append((log_pos, self._format_fft_frequency(freq)))
        
        # Setze Ticks auf X-Achse
        x_axis = self.fft_plot.getAxis('bottom')
        x_axis.setTicks([ticks])
    
    def _open_file(self):
        """Datei öffnen Dialog."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Audiodatei öffnen", "",
            "Audio (*.wav *.mp3);;Alle Dateien (*)"
        )
        if filename:
            self._load_file(filename)
    
    def _load_file(self, filepath: str):
        """Audiodatei laden."""
        try:
            self.status_label.setText(f"Lade {Path(filepath).name}...")
            QApplication.processEvents()
            
            self._audio = load_audio(filepath)
            
            # Waveform anzeigen
            data = self._audio.get_channel(0)  # Linker Kanal / Mono
            time = np.arange(len(data)) / self._audio.sample_rate
            
            # Downsampling für Anzeige
            if len(data) > 50000:
                factor = len(data) // 50000
                data_ds = data[::factor]
                time_ds = time[::factor]
                self.waveform_curve.setData(time_ds, data_ds)
            else:
                self.waveform_curve.setData(time, data)
            
            # Achsen setzen
            self.waveform_plot.setXRange(0, self._audio.duration_seconds)
            self.waveform_plot.setYRange(-1, 1)
            
            # Selektion auf ganzen Bereich + Grenzen setzen
            self.selection_region.setBounds([0, self._audio.duration_seconds])
            self.selection_region.setRegion([0, self._audio.duration_seconds])
            
            # Cursor zurücksetzen
            self.playback_cursor.setValue(0)
            self.spectro_cursor.setValue(0)
            
            # Spektrogramm berechnen
            self._update_spectrogram()
            
            # UI updaten
            self.file_label.setText(
                f"{self._audio.file_path.name} | "
                f"{self._audio.sample_rate} Hz | "
                f"{'Stereo' if self._audio.channels == 2 else 'Mono'} | "
                f"{format_time(self._audio.duration_seconds)}"
            )
            self.btn_analyze.setEnabled(True)
            self.btn_play_selection.setEnabled(True)
            self.status_label.setText("")
            self._bands = []
            self.band_list.clear()
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Konnte Datei nicht laden:\n{e}")
    
    def _create_acoustic_colormap(self):
        """Erstelle klassische Spektrogramm-Colormap (jet-ähnlich) wie in pyqtgraph-spectrographer."""
        # Versuche zuerst die eingebaute "jet" Colormap von pyqtgraph zu verwenden
        try:
            cmap = pg.colormap.get('jet')
            self.spectro_img.setColorMap(cmap)
        except:
            # Fallback: Benutzerdefinierte "jet"-ähnliche Colormap
            # Klassische "jet" Colormap für Spektrogramme: blau -> cyan -> grün -> gelb -> rot
            # Optimiert für akustische Analysen mit guter Frequenzauflösung
            colors = [
                (0.0, (0, 0, 128)),         # Dunkelblau (sehr leise)
                (0.25, (0, 0, 255)),        # Blau (leise)
                (0.5, (0, 255, 255)),       # Cyan (mittel)
                (0.75, (255, 255, 0)),      # Gelb (laut)
                (1.0, (255, 0, 0)),         # Rot (sehr laut)
            ]
            
            positions = [c[0] for c in colors]
            rgb_colors = [c[1] for c in colors]
            
            cmap = pg.ColorMap(positions, rgb_colors)
            self.spectro_img.setColorMap(cmap)
    
    def _update_spectrogram(self):
        """Spektrogramm berechnen und anzeigen."""
        if self._audio is None:
            return
        
        data = self._audio.get_channel(0)
        # Sehr hohe Auflösung für akustische Analysen: 16384 FFT, 95% Overlap
        config = SpectrogramConfig(fft_size=16384, overlap_percent=95.0)
        result = compute_spectrogram(data, self._audio.sample_rate, config)
        
        # dB konvertieren mit erweitertem Bereich für bessere Dynamik
        db_min = -100  # Erweiterter Bereich für bessere Sichtbarkeit
        db_max = 0
        magnitude_db = result.magnitude_db(min_db=db_min)
        
        # Auf uint8 für pyqtgraph konvertieren (0-255) mit erweitertem dB-Bereich
        img_data = (magnitude_db - db_min) / (db_max - db_min)  # 0-1
        img_data = np.clip(img_data, 0, 1)
        img_uint8 = (img_data * 255).astype(np.uint8)
        
        # Image setzen (transponiert: Zeit auf X, Frequenz auf Y)
        self.spectro_img.setImage(img_uint8.T)
        
        # Skalierung auf echte Zeit/Frequenz-Achsen
        time_max = result.times[-1] if len(result.times) > 0 else 1
        freq_max = result.frequencies[-1] if len(result.frequencies) > 0 else 1
        
        # Transform für korrekte Achsenskalierung
        tr = pg.QtGui.QTransform()
        tr.scale(time_max / img_uint8.shape[1], freq_max / img_uint8.shape[0])
        self.spectro_img.setTransform(tr)
        
        # Levels für bessere Darstellung setzen
        self.spectro_img.setLevels([0, 255])
        
        self.spectro_plot.setXRange(0, self._audio.duration_seconds)
        self.spectro_plot.setYRange(0, min(self._audio.sample_rate / 2, 20000))
    
    def _on_selection_changed(self):
        """Selektion geändert."""
        if self._audio is None:
            return
        
        region = self.selection_region.getRegion()
        self._selection_start = max(0, region[0])
        self._selection_end = min(self._audio.duration_seconds, region[1])
        
        duration = self._selection_end - self._selection_start
        start_samples = int(self._selection_start * self._audio.sample_rate)
        end_samples = int(self._selection_end * self._audio.sample_rate)
        
        self.selection_label.setText(
            f"Selektion: {format_time(self._selection_start)} – "
            f"{format_time(self._selection_end)} "
            f"({format_time(duration)} | {end_samples - start_samples:,} Samples)"
        )
        
        # Play-Button aktivieren wenn gültige Selektion
        if self._audio is not None and duration > 0.01:
            self.btn_play_selection.setEnabled(True)
        else:
            self.btn_play_selection.setEnabled(False)
        
        # FFT aktualisieren wenn Tab aktiv
        if self.tabs.currentIndex() == 1:
            self._update_fft()
    
    def _on_tab_changed(self, index: int):
        """Tab gewechselt."""
        if index == 1:  # FFT Tab
            self._update_fft()
    
    def _sync_from_waveform(self):
        """Waveform-Zoom → Spektrogramm synchronisieren."""
        if self._syncing or self._audio is None:
            return
        self._syncing = True
        view_range = self.waveform_plot.viewRange()
        self.spectro_plot.setXRange(view_range[0][0], view_range[0][1], padding=0)
        self._syncing = False
    
    def _sync_from_spectro(self):
        """Spektrogramm-Zoom → Waveform synchronisieren."""
        if self._syncing or self._audio is None:
            return
        self._syncing = True
        view_range = self.spectro_plot.viewRange()
        self.waveform_plot.setXRange(view_range[0][0], view_range[0][1], padding=0)
        self._syncing = False
    
    def _update_fft(self):
        """FFT für selektierten Bereich berechnen."""
        if self._audio is None:
            return
        
        # Selektierten Bereich holen
        start_sample = int(self._selection_start * self._audio.sample_rate)
        end_sample = int(self._selection_end * self._audio.sample_rate)
        data = self._audio.get_channel(0)[start_sample:end_sample]
        
        if len(data) < 64:
            self.fft_info.setText("Selektion zu kurz für FFT")
            return
        
        # FFT Parameter
        fft_size = int(self.fft_size_combo.currentText())
        window_name = self.window_combo.currentText()
        
        # Fenster
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
        
        # Frequenzachse
        freqs = np.fft.rfftfreq(fft_size, 1 / self._audio.sample_rate)
        
        # Plot
        self.fft_curve.setData(freqs[1:], magnitude_db[1:])  # Ohne DC
        f_max = min(self._audio.sample_rate / 2, 16000)  # Max 16 kHz für Audio
        self.fft_plot.setXRange(np.log10(20), np.log10(f_max))
        self.fft_plot.setYRange(-80, 0)
        
        # Professionelle Frequenzmarkierungen aktualisieren
        self._update_fft_axis_ticks(f_max)
        
        # Info
        self.fft_info.setText(
            f"Samples: {len(data):,} | "
            f"Frequenzauflösung: {self._audio.sample_rate / fft_size:.2f} Hz"
        )
    
    def _start_analysis(self):
        """Terzbandanalyse starten."""
        if self._audio is None:
            return
        
        # Selektierten Bereich holen
        start_sample = int(self._selection_start * self._audio.sample_rate)
        end_sample = int(self._selection_end * self._audio.sample_rate)
        data = self._audio.get_channel(0)[start_sample:end_sample]
        
        if len(data) < 1000:
            QMessageBox.warning(self, "Hinweis", "Selektion zu kurz für Terzbandanalyse")
            return
        
        # UI
        self.btn_analyze.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.status_label.setText("Berechne Terzbänder...")
        
        # Worker starten
        self._worker = FilterWorker(data, self._audio.sample_rate)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.start()
    
    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        """Fortschritt aktualisieren."""
        self.progress.setMaximum(total)
        self.progress.setValue(current)
    
    @Slot(list)
    def _on_analysis_done(self, bands: list):
        """Analyse fertig."""
        self._bands = bands
        self.btn_analyze.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.progress.hide()
        self.status_label.setText(f"Analyse abgeschlossen - {len(bands)} Bänder")
        
        # Liste füllen
        self.band_list.clear()
        for band in bands:
            text = f"{format_frequency(band.center_frequency)}"
            self.band_list.addItem(text)
        
        if bands:
            self.band_list.setCurrentRow(0)
    
    def _on_band_selected(self, row: int):
        """Terzband ausgewählt."""
        if row < 0 or row >= len(self._bands):
            return
        
        band = self._bands[row]
        
        # Zeit
        time = np.arange(len(band.filtered_signal)) / band.sample_rate
        
        # Signal plotten
        self.impulse_curve.setData(time, band.filtered_signal)
        
        # Hüllkurve
        envelope = band.envelope(method="hilbert")
        self.envelope_curve.setData(time, envelope)
        
        # Cursor zurücksetzen (wenn nicht gerade abgespielt wird)
        if not self._is_playing_band:
            self.band_cursor.setValue(0)
        
        # Titel
        self.impulse_plot.setTitle(
            f"Terzband {format_frequency(band.center_frequency)} | "
            f"RMS: {format_db(band.rms_db())}"
        )
    
    def _play_band(self):
        """Ausgewähltes Band abspielen."""
        row = self.band_list.currentRow()
        if row < 0 or row >= len(self._bands):
            return
        
        band = self._bands[row]
        
        # Stoppe vorherige Wiedergabe
        self._stop_playback()
        
        try:
            import sounddevice as sd
            import time
            
            # Stop falls schon was läuft
            sd.stop()
            
            # Abspielen
            self._band_stream = sd.play(band.filtered_signal.astype(np.float32), band.sample_rate)
            self._band_playback_start_time = time.time()
            self._is_playing_band = True
            self._current_band_duration = len(band.filtered_signal) / band.sample_rate
            
            # Cursor auf Startposition setzen
            self.band_cursor.setValue(0)
            self.band_cursor.setVisible(True)
            
            # Timer für Cursor-Update starten
            if self._band_timer is None:
                self._band_timer = QTimer()
                self._band_timer.timeout.connect(self._update_band_cursor)
            self._band_timer.start(50)  # 20 FPS - ausreichend flüssig, weniger CPU-Last
            
            self.btn_stop.setEnabled(True)
            self.status_label.setText(f"▶ Spiele {format_frequency(band.center_frequency)}...")
            
        except Exception as e:
            QMessageBox.warning(self, "Wiedergabe-Fehler", str(e))
            self._stop_playback()
    
    def _stop_playback(self):
        """Wiedergabe stoppen."""
        self._is_playing_band = False
        
        try:
            import sounddevice as sd
            sd.stop()
            self._band_stream = None
        except:
            pass
        
        # Timer stoppen
        if self._band_timer:
            self._band_timer.stop()
        
        # Cursor zurücksetzen und verstecken
        self.band_cursor.setValue(0)
        self.band_cursor.setVisible(False)
        
        self.btn_stop.setEnabled(False)
        self.status_label.setText("")
    
    def _update_band_cursor(self):
        """Cursor während Terzband-Wiedergabe aktualisieren."""
        if not self._is_playing_band:
            return
        
        try:
            import time
            
            # Position basierend auf verstrichener Zeit berechnen
            elapsed = time.time() - self._band_playback_start_time
            current_time = elapsed
            
            # Cursor aktualisieren
            if current_time <= self._current_band_duration:
                self.band_cursor.setValue(current_time)
            else:
                # Ende erreicht - automatisch stoppen
                self._stop_playback()
                
        except Exception as e:
            # Bei Fehler stoppen
            print(f"Band cursor update error: {e}")
            self._stop_playback()
    
    def _play_selection(self):
        """Selektierten Bereich abspielen."""
        if self._audio is None:
            return
        
        # Stoppe vorherige Wiedergabe
        self._stop_selection_playback()
        
        try:
            import sounddevice as sd
            
            # Selektierten Bereich holen
            start_sample = int(self._selection_start * self._audio.sample_rate)
            end_sample = int(self._selection_end * self._audio.sample_rate)
            
            if end_sample <= start_sample:
                return
            
            # Daten extrahieren
            if self._audio.channels == 1:
                data = self._audio.data[start_sample:end_sample]
            else:
                # Stereo: beide Kanäle
                data = self._audio.data[start_sample:end_sample]
            
            # Abspielen
            import time
            self._selection_stream = sd.play(data.astype(np.float32), self._audio.sample_rate)
            self._selection_playback_start_time = time.time()
            self._is_playing_selection = True
            
            # Cursor auf Startposition setzen
            self.playback_cursor.setValue(self._selection_start)
            self.spectro_cursor.setValue(self._selection_start)
            self.playback_cursor.setVisible(True)
            self.spectro_cursor.setVisible(True)
            
            # Timer für Cursor-Update starten
            if self._selection_timer is None:
                self._selection_timer = QTimer()
                self._selection_timer.timeout.connect(self._update_playback_cursor)
            self._selection_timer.start(50)  # 20 FPS - ausreichend flüssig, weniger CPU-Last
            
            # UI aktualisieren
            self.btn_play_selection.setEnabled(False)
            self.btn_stop_selection.setEnabled(True)
            duration = self._selection_end - self._selection_start
            self.status_label.setText(f"▶ Wiedergabe selektierter Bereich ({format_time(duration)})")
            
        except Exception as e:
            QMessageBox.warning(self, "Wiedergabe-Fehler", str(e))
            self._stop_selection_playback()
    
    def _stop_selection_playback(self):
        """Wiedergabe des selektierten Bereichs stoppen."""
        self._is_playing_selection = False
        
        try:
            import sounddevice as sd
            sd.stop()
            self._selection_stream = None
        except:
            pass
        
        # Timer stoppen
        if self._selection_timer:
            self._selection_timer.stop()
        
        # Cursor zurücksetzen und verstecken
        if self._audio is not None:
            self.playback_cursor.setValue(self._selection_start)
            self.spectro_cursor.setValue(self._selection_start)
        self.playback_cursor.setVisible(False)
        self.spectro_cursor.setVisible(False)
        
        # UI aktualisieren
        self.btn_play_selection.setEnabled(True)
        self.btn_stop_selection.setEnabled(False)
        self.status_label.setText("")
    
    def _update_playback_cursor(self):
        """Cursor während Wiedergabe aktualisieren."""
        if not self._is_playing_selection or self._audio is None:
            return
        
        try:
            import time
            
            # Position basierend auf verstrichener Zeit berechnen
            elapsed = time.time() - self._selection_playback_start_time
            current_time = self._selection_start + elapsed
            
            # Cursor aktualisieren (synchron in beiden Plots)
            if current_time <= self._selection_end:
                self.playback_cursor.setValue(current_time)
                self.spectro_cursor.setValue(current_time)
            else:
                # Ende des selektierten Bereichs erreicht - automatisch stoppen
                self._stop_selection_playback()
                
        except Exception as e:
            # Bei Fehler stoppen
            print(f"Cursor update error: {e}")
            self._stop_selection_playback()
    
    def _export_band(self):
        """Ausgewähltes Band exportieren."""
        row = self.band_list.currentRow()
        if row < 0 or row >= len(self._bands):
            return
        
        band = self._bands[row]
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Terzband exportieren",
            f"terzband_{int(band.center_frequency)}Hz.wav",
            "WAV (*.wav)"
        )
        
        if filename:
            try:
                save_audio(band.filtered_signal, filename, band.sample_rate)
                QMessageBox.information(self, "Export", f"Exportiert nach:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", str(e))
    
    def dragEnterEvent(self, event):
        """Drag & Drop."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.wav', '.mp3')):
                    event.acceptProposedAction()
                    return
    
    def dropEvent(self, event):
        """Datei gedroppt."""
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            if filepath.lower().endswith(('.wav', '.mp3')):
                self._load_file(filepath)
                break
    
    def closeEvent(self, event):
        """Beim Schließen."""
        # Wiedergabe stoppen
        self._stop_playback()
        self._stop_selection_playback()
        
        # Timer stoppen
        if self._selection_timer:
            self._selection_timer.stop()
        if self._band_timer:
            self._band_timer.stop()
        
        # Worker beenden
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        event.accept()
