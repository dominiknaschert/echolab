# Audio Analyzer

## Analysetool von Audiodateien mit Fokus auf zeitabhängige Terzband-Impulsantworten zur besseren Diagnose von Flatterechos.

Das tool dient der bestimmung von Flatterechos aus Audioaufzeichnungen. Für die eingelesene Audiodatei können trezband-Impulsantworten (Audio/Zeitverlauf) erstellt werden. Mithilfe deiser kann einfach die Terzbänder des Flatterechos bestimmt werden. Die Terzbandimpulsantowrten können als wav exportiert werden.

## Installation

### Entwicklungsumgebung

```bash
# Python 3.11+ erforderlich
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oder: venv\Scripts\activate  # Windows

pip install -r requirements.txt
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
