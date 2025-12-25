# Introduction

**Echolab** is an analysis tool for the detection of flutter echoes. It uses room impulse responses to visualize flutter echoes in the time domain. In addition to the visual time-domain analysis, the tool implements the detection algorithm presented by Schanda et al. (DAGA 2023). The algorithm identifies repetition frequencies using autocorrelation and calculates the corresponding distances.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Time-domain analysis** with amplitude evolution visualization and parallel spectrogram display
- **FFT analysis** with configurable window functions and FFT sizes
- **Third-octave-band impulse responses** generation from imported audio files to identify frequency bands where flutter echoes occur
- **Audio playback** and **WAV export** of third-octave-band impulse responses
- **Flutter echo detection** using the method by Schanda et al. (DAGA 2023)

## GUI 

<div>
  <img src="https://github.com/user-attachments/assets/3a00a6c7-4f7b-49f3-b739-c232fa1f1e25" width="49%" alt="Time Domain" />
  <img src="https://github.com/user-attachments/assets/c79e5253-e4c4-44d7-9824-23a82a63a99f" width="49%" alt="FFT" />
</div>
<div>
  <img src="https://github.com/user-attachments/assets/c54e6ade-30c7-4304-a4b2-53dafc3c16e0" width="49%" alt="Third-Octave" />
  <img src="https://github.com/user-attachments/assets/cc1efe70-d0e1-4928-9585-ed13896ef194" width="49%" alt="Room Analysis" />
</div>


## Usage

1. **Load an audio file**: Use File → Open or drag and drop a WAV/MP3 file
2. **Select a region**: Click and drag in the time domain view to select the relevant impulse response region for the further analysis.
3. **Analyze**:
   - **FFT Analysis**: Switch to the FFT tab to view frequency spectrum
   - **Third-Octave Bands**: Use the Third-Octave tab to generate filtered impulse responses
   - **Room Analysis**: Click "Analyze" in the Room Analysis tab to detect flutter echoes
  
### Windows Executable

```bash
pyinstaller echolab.specThe 
```
executable will be created in `dist/echolab.exe`.

## Primary References

The detection algorithm is based on:
- **Schanda, J., Hoffbauer, P., & Lachenmayr, G.** (2023). *Flutter Echo Detection in Room Impulse Responses*. DAGA 2023.

## Supporting Literatur 
- **Halmrast, T.** (2015). "Why Do Flutter Echos Always End Up Around 1-2 kHz?" In: Proceedings of the Institute of Acoustics, Vol. 37 Pt. 3, pp. 395-408.
- **Kuhl, W.** (1984). Nachhallzeiten schwach gedämpfter geschlossener Wellenzüge. Acustica 55, pp. 187-192.
- **Lorenz-Kierakiewitz, K.-H.** (2019). Flatterechos und wo sie zu finden sind. In: Fortschritte der Akustik - DAGA 2019.
- **Rindel, J. H.** (2016). Detection of Colouration in Rooms by use of Cepstrum Technique. In: Fortschritte der Akustik - DAGA 2016.
- **Yamada, Y., et al.** (2006). A simple method to detect audible echoes in room acoustical design. Applied Acoustics 67(9), pp. 835-848.

## License

[MIT](https://choosealicense.com/licenses/mit/)
