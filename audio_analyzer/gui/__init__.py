"""
GUI-Modul für Audio Analyzer.

Verwendet PySide6 und pyqtgraph für interaktive Visualisierung.
Strikte Trennung von DSP-Logik - dieses Modul enthält nur Darstellung.
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]

