Audio Analyzer

Audio file analysis tool with a focus on time-dependent third-octave-band impulse responses for improved diagnosis of flutter echoes.

The tool is designed to detect flutter echoes from audio recordings. For an imported audio file, third-octave-band impulse responses can be generated. These allow straightforward identification of the third-octave bands in which flutter echoes occur. The third-octave-band impulse responses can be listend back to or exported as WAV files.

<img width="994" height="736" alt="Image" src="https://github.com/user-attachments/assets/2335b1ac-ab21-4697-87bb-a3e05fa803fb" />

<img width="999" height="738" alt="Image" src="https://github.com/user-attachments/assets/94a48b2e-38ba-4bc7-b449-a2bd680be28e" />

<img width="996" height="736" alt="Image" src="https://github.com/user-attachments/assets/82eebb82-a4db-4030-ab79-07c1a3e71931" />

## Installation

### Development Environment

```bash
# Python 3.11+ required
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Creating a Windows Executable

```bash
pyinstaller audio_analyzer.spec
# Output in dist/AudioAnalyzer.exe
```

Technical Documentation

Third-Octave Filter Bank (IEC 61260)

The filter bank uses 6th-order IIR Butterworth filters with:
	•	Standardized center frequencies according to IEC 61260-1
	•	Bandwidth: fm × (2^(1/6) − 2^(−1/6))
	•	Documented phase behavior and group delay

Dependencies
	•	Python 3.11+
	•	PyQt6 (GUI)
	•	numpy, scipy (signal processing)
	•	pyqtgraph (visualization)
	•	soundfile (WAV I/O)
	•	librosa / audioread (audio decoding)
	•	sounddevice (playback)

License

MIT License

