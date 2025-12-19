# Technische Annahmen und Dokumentation

Dieses Dokument beschreibt alle technischen Annahmen, Vereinfachungen und Design-Entscheidungen der Audio Analyzer Anwendung.

## Inhaltsverzeichnis

1. [Audio I/O](#audio-io)
2. [Signalverarbeitung](#signalverarbeitung)
3. [Spektrogramm](#spektrogramm)
4. [Terzbandfilterbank](#terzbandfilterbank)
5. [Impulsantwort-Berechnung](#impulsantwort-berechnung)
6. [Architektur-Entscheidungen](#architektur-entscheidungen)

---

## Audio I/O

### Dateiformat-Unterstützung

| Format | Bibliothek | Anmerkung |
|--------|------------|-----------|
| WAV | soundfile (libsndfile) | Alle PCM-Subtypes (8/16/24/32-bit, float) |
| MP3 | librosa + audioread | Verlustbehaftete Dekodierung |

### Annahmen

1. **Keine automatische Konvertierung**
   - Samplerate wird nicht automatisch geändert
   - Kanäle werden nicht automatisch zusammengemischt
   - Der Nutzer muss explizit konvertieren

2. **Datenformat**
   - Alle Audiodaten werden intern als `float64` gespeichert
   - Wertebereich: -1.0 bis +1.0
   - Stereo-Format: `(samples, 2)` mit Kanal 0 = Links, Kanal 1 = Rechts

3. **MP3-Dekodierung**
   - MP3 ist verlustbehaftet – Originaldaten sind nicht rekonstruierbar
   - `librosa.load(sr=None)` verhindert Resampling auf 22050 Hz
   - Bit-Tiefe ist bei MP3 nicht definiert

### Vereinfachungen

- Keine Unterstützung für mehrkanalige Formate (> 2 Kanäle)
- Keine Unterstützung für exotische WAV-Subtypes (A-law, µ-law)
- Keine Metadaten-Extraktion (ID3-Tags, etc.)

---

## Signalverarbeitung

### Resampling

**Implementierung:** `scipy.signal.resample_poly`

**Eigenschaften:**
- Polyphasen-FIR-Filter für effiziente Berechnung
- Automatische Anti-Aliasing-Filterung (Kaiser-Fenster)
- Gruppenlaufzeit wird durch Padding kompensiert

**Annahmen:**
- Samplerate-Verhältnis wird auf ganzzahlige Faktoren reduziert (GCD)
- Keine manuelle Filterdesign-Optionen exponiert

### Downmix

**Methoden:**

| Methode | Formel | Anwendung |
|---------|--------|-----------|
| average | (L + R) / 2 | Standard-Downmix |
| left | L | Nur linker Kanal |
| right | R | Nur rechter Kanal |
| mid | (L + R) / 2 | Identisch zu average |
| side | (L - R) / 2 | Stereodifferenz |

**Wichtiger Hinweis zur Energiekompensation:**

Der `average`-Downmix dividiert durch 2, was bei korreliertem Material (z.B. zentrierte Signale) zu einer Pegelreduktion von ~6 dB führen kann. Diese Kompensation wird **nicht automatisch** angewendet, da:

1. Die Korrelation signalabhängig ist
2. Bei unkorreliertem Material (z.B. Stereo-Rauschen) würde Kompensation zu Übersteuerung führen
3. Für reproduzierbare Analyse ist der unkompensierte Downmix transparenter

---

## Spektrogramm

### STFT-Parameter

**Implementierung:** `scipy.signal.stft`

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| FFT-Größe | 2048 | Frequenzauflösung: fs/N Hz |
| Überlappung | 75% | Zeitauflösung vs. Redundanz |
| Fensterfunktion | Hann | Guter Kompromiss |
| Padding | Zero-Padding | An Rändern |

### Heisenberg-Unsicherheit

Die Frequenz-Zeit-Auflösung unterliegt einer fundamentalen Einschränkung:

```
Δf × Δt ≥ 1/(4π)
```

Bei 44100 Hz Samplerate:

| FFT-Größe | Δf (Hz) | Δt (ms) bei 75% Overlap |
|-----------|---------|-------------------------|
| 512 | 86.1 | 2.9 |
| 1024 | 43.1 | 5.8 |
| 2048 | 21.5 | 11.6 |
| 4096 | 10.8 | 23.2 |
| 8192 | 5.4 | 46.4 |

**Empfehlung:** FFT 2048 für allgemeine Analyse, 4096+ für tiefe Frequenzen.

### Fensterfunktionen

| Fenster | Seitenkeulendämpfung | Hauptkeulenbreite | Anwendung |
|---------|----------------------|-------------------|-----------|
| Rectangular | -13 dB | Schmalste | Impulsanalyse |
| Hann | -31.5 dB | Mittel | Standard |
| Hamming | -43 dB | Mittel | Ältere Systeme |
| Blackman | -58 dB | Breit | Hohe Dynamik |
| Kaiser | Einstellbar | Einstellbar | Speziell |

---

## Terzbandfilterbank

### IEC 61260-1:2014 Konformität

Die Filterbank implementiert die Anforderungen der IEC 61260-1:2014 für 1/3-Oktav-Filter.

### Mittenfrequenzen

Basierend auf der Referenzfrequenz 1000 Hz:

```
f_m = 1000 × 10^(n/10) Hz
```

Für n = -18 bis +13 ergeben sich Mittenfrequenzen von 12.5 Hz bis 20 kHz.

**Implementierte Frequenzen:**
12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 Hz

### Bandgrenzen

Für 1/3-Oktav-Filter:

```
f_lower = f_m / 2^(1/6) ≈ f_m × 0.891
f_upper = f_m × 2^(1/6) ≈ f_m × 1.122
```

Bandbreite:
```
B = f_m × (2^(1/6) - 2^(-1/6)) ≈ 0.2316 × f_m
```

### Filterdesign

**Typ:** Butterworth IIR (Infinite Impulse Response)

**Ordnung:** 6 (entspricht Class 1 Anforderungen)

**Implementierung:** Second-Order Sections (SOS) für numerische Stabilität

### Phasenverhalten

**IIR-Filter haben nichtlineares Phasenverhalten!**

Dies bedeutet:
1. Gruppenlaufzeit variiert mit Frequenz
2. Impulsantworten sind asymmetrisch
3. Zeitliche Lokalisierung kann verfälscht sein

**Gruppenlaufzeit bei Mittenfrequenz (typisch):**

| f_m (Hz) | τ_g (Samples @ 44.1kHz) | τ_g (ms) |
|----------|-------------------------|----------|
| 100 | ~1500 | ~34 |
| 1000 | ~150 | ~3.4 |
| 10000 | ~15 | ~0.34 |

### Zero-Phase-Filterung (Optional)

Mit `filtfilt` (Vorwärts-Rückwärts-Filterung):

**Vorteile:**
- Lineare Phase (keine Gruppenläufzeit-Verzerrung)
- Symmetrische Impulsantworten
- Doppelte effektive Filterordnung (12)

**Nachteile:**
- Nicht kausal (für Offline-Analyse akzeptabel)
- Doppelter Rechenaufwand
- Artefakte an Signalrändern

---

## Impulsantwort-Berechnung

### Methodik

Für jedes Terzband wird die zeitabhängige Energie berechnet:

1. **Filterung:** Signal durch Terzband-Filter
2. **Hüllkurve:** Berechnung der Momentanleistung
3. **Zeitauflösung:** Segmentierung in Zeitfenster

### Hüllkurven-Berechnung

**Methode 1: Hilbert-Transformation (Standard)**

```python
analytic_signal = scipy.signal.hilbert(filtered)
envelope = np.abs(analytic_signal)
```

- Präzise instantane Amplitude
- Kein Fensterparameter
- Kann bei niedrigen Frequenzen Artefakte haben

**Methode 2: Gleitender RMS**

```python
window_size = int(sample_rate / center_freq * 2)
envelope = sliding_rms(filtered, window_size)
```

- Robuster
- Glättet hochfrequente Komponenten
- Fenstergröße beeinflusst Zeitauflösung

### Zeitauflösung

Die Standard-Zeitauflösung von 10 ms ermöglicht:
- 100 Frames pro Sekunde
- Auflösung von Transienten > 20 ms
- Kompromiss zwischen Detail und Datenmenge

Für Impulsantwort-Analyse in Räumen:
- 1-5 ms für frühe Reflexionen
- 10-50 ms für späten Nachhall

---

## Architektur-Entscheidungen

### Strikte Trennung

```
┌─────────────────────────────────────────────────────┐
│                    GUI Layer                         │
│  (PyQt6 Widgets, pyqtgraph Visualisierung)          │
├─────────────────────────────────────────────────────┤
│                 Application Layer                    │
│  (MainWindow, Signal-Koordination)                   │
├─────────────────────────────────────────────────────┤
│                   Core Layer                         │
│  (DSP, Filterbank, I/O - ohne GUI-Dependencies)     │
└─────────────────────────────────────────────────────┘
```

**Prinzipien:**

1. **Core hat keine GUI-Imports**
   - `audio_analyzer.core` importiert niemals PyQt6
   - Alle DSP-Funktionen sind mit numpy/scipy testbar

2. **GUI ist reine Darstellung**
   - Widgets zeigen Daten an, berechnen nicht
   - Schwere Berechnungen in Worker-Threads

3. **Explizite Datenflüsse**
   - Keine globalen Zustände
   - Daten werden explizit übergeben

### Threading-Modell

- Filterbank-Berechnung in `QThread`
- GUI bleibt responsiv
- Progress-Signale für Statusanzeige

### Keine impliziten Manipulationen

**Explizit erlaubt:**
- Downmix nach Nutzer-Bestätigung
- Resampling nach Nutzer-Bestätigung
- Normalisierung nach Nutzer-Bestätigung

**Verboten:**
- Automatisches Resampling beim Laden
- Automatischer Downmix
- Stille Clip-Korrektur
- Automatische Pegelanpassung

---

## Nicht implementierte Features

Bewusst **nicht** Teil dieses Projekts:

1. **Echtzeit-Verarbeitung** – Nur Offline-Analyse
2. **Mehrkanal > 2** – Maximal Stereo
3. **VST/Plugin-Support** – Keine externe Plugin-Schnittstelle
4. **Machine Learning** – Keine automatische Interpretation
5. **A/C-Gewichtung** – Keine psychoakustische Bewertung
6. **Kalibrierung** – Keine Referenzpegel-Einstellung
7. **Raumakustik-Berechnungen** – Kein RT60, EDT, etc.

---

## Validierung

Die Filterbank wurde gegen folgende Referenzen validiert:

1. **Sinustöne bei Mittenfrequenzen** – Durchlass > 80%
2. **Sinustöne 2 Oktaven entfernt** – Dämpfung > 30 dB
3. **Bandbreiten-Messung** – Innerhalb ±5% der IEC-Spezifikation
4. **Weißes Rauschen** – Flaches Spektrum ±3 dB

---

*Letzte Aktualisierung: 2024*

