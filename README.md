# Audio Analyzer

## Analysetool von Audiodateien mit Fokus auf zeitabhängige Terzband-Impulsantworten zur besseren Diagnose von Flatterechos.

Das tool dient der bestimmung von Flatterechos aus Audioaufzeichnungen. Für die eingelesene Audiodatei können trezband-Impulsantworten (Audio/Zeitverlauf) erstellt werden. Mithilfe deiser kann einfach die Terzbänder des Flatterechos bestimmt werden. Die Terzbandimpulsantowrten können als wav exportiert werden.

<img width="999" height="743" alt="Image" src="https://github.com/user-attachments/assets/e2f0a35a-0750-4206-9c23-de9f23bf8c5e" />

<img width="1000" height="741" alt="Image" src="https://github.com/user-attachments/assets/e66e3d35-606f-4dd1-9163-94859dd421e3" />

<img width="998" height="746" alt="Image" src="https://github.com/user-attachments/assets/1fbc9a4b-2dd3-45ca-bbf0-31ce16de6fce" />

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
