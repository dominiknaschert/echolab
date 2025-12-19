# Audio Analyzer

## Offline-Analyse von Audiodateien mit Fokus auf zeitabhängige Terzband-Impulsantworten

Diese Python-Desktop-Applikation dient der präzisen Analyse von Audiodateien. Der Schwerpunkt liegt auf der normgerechten Terzbandanalyse nach IEC 61260 und der Berechnung zeitabhängiger Impulsantworten pro Terzband.

## Features

- **Waveform-Darstellung**: Zoombar mit präziser Bereichsmarkierung
- **Spektrogramm**: Logarithmische Frequenzachse, konfigurierbare FFT-Parameter
- **Zeitselektion**: Nicht-destruktive Analysefenster (Zeit + Samples)
- **Terzbandanalyse**: 1/3-Oktav-Filterbank nach IEC 61260
- **Impulsantworten**: Zeitabhängige Darstellung pro Terzband
- **Export**: Einzelne Terzband-Impulsantworten als WAV

## Technische Architektur

```
audio_analyzer/
├── core/           # DSP-Logik (ohne GUI-Abhängigkeiten)
│   ├── audio_io.py       # Audio-Import/Export
│   ├── signal_processing.py  # Allgemeine Signalverarbeitung
│   ├── spectral.py       # Spektrogramm-Berechnung
│   └── third_octave.py   # IEC 61260 Terzbandfilterbank
├── gui/            # PyQt6 GUI-Komponenten
│   ├── main_window.py
│   ├── waveform_widget.py
│   ├── spectrogram_widget.py
│   ├── third_octave_widget.py
│   └── dialogs/
├── utils/          # Hilfsfunktionen
└── main.py         # Einstiegspunkt
```

### Strikte Trennung

- **DSP-Kern**: Vollständig ohne GUI testbar
- **GUI-Logik**: Reine Darstellung und Benutzerinteraktion
- **I/O**: Separate Schicht für Dateioperationen

## Installation

### Entwicklungsumgebung

```bash
# Python 3.11+ erforderlich
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oder: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Anwendung starten

```bash
python main.py
```

### Windows-Executable erstellen

```bash
pyinstaller audio_analyzer.spec
# Ergebnis in dist/AudioAnalyzer.exe
```

## Technische Dokumentation

### Terzbandfilterbank (IEC 61260)

Die Filterbank verwendet **IIR Butterworth-Filter 6. Ordnung** mit:
- Normgerechten Mittenfrequenzen nach IEC 61260-1
- Bandbreite: fm × (2^(1/6) - 2^(-1/6))
- Dokumentiertes Phasenverhalten und Gruppenlaufzeit

**Hinweis**: IIR-Filter haben nichtlineares Phasenverhalten. Für phasenkritische Anwendungen 
ist Zero-Phase-Filtering (filtfilt) optional aktivierbar.


## Abhängigkeiten

- Python 3.11+
- PyQt6 (GUI)
- numpy, scipy (Signalverarbeitung)
- pyqtgraph (interaktive Visualisierung)
- soundfile (WAV-I/O)
- librosa/audioread (MP3-Dekodierung)
- sounddevice (Wiedergabe)

## Lizenz

MIT License
