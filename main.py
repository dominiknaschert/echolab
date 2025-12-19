#!/usr/bin/env python3
"""
Audio Analyzer - Einstiegspunkt

Offline-Analyse von Audiodateien mit Fokus auf zeitabhängige Terzband-Impulsantworten.

Verwendung:
    python main.py [audio_file]

Beispiel:
    python main.py recording.wav
"""

import sys
from pathlib import Path


def main():
    """Start the Audio Analyzer application."""
    # Check Python version
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    # Import PySide6 (late import for faster error if not installed)
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print("Error: PySide6 is not installed.")
        print("Install with: pip install PySide6")
        sys.exit(1)
    
    # Import our application
    from audio_analyzer.gui import MainWindow
    
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Analyzer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AudioAnalyzer")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Load file if provided as argument
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        if filepath.exists() and filepath.suffix.lower() in ('.wav', '.mp3'):
            window._load_audio_file(str(filepath))
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

