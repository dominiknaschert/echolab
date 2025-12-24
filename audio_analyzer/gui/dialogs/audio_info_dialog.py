"""
Audio Info Dialog

Displays detailed information about a loaded audio file.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QFormLayout, QPushButton,
)

from ...core.audio_io import AudioFile
from ...utils.formatting import format_time, format_sample_rate, format_channels


class AudioInfoDialog(QDialog):
    """Dialog showing detailed audio file information."""
    
    def __init__(self, audio: AudioFile, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Information")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # File info
        file_group = QGroupBox("File")
        file_layout = QFormLayout(file_group)
        file_layout.addRow("Filename:", QLabel(audio.file_path.name))
        file_layout.addRow("Path:", QLabel(str(audio.file_path.parent)))
        layout.addWidget(file_group)
        
        # Format info
        format_group = QGroupBox("Format")
        format_layout = QFormLayout(format_group)
        format_layout.addRow("Sample Rate:", QLabel(format_sample_rate(audio.sample_rate)))
        format_layout.addRow("Channels:", QLabel(format_channels(audio.channels)))
        
        if audio.bit_depth:
            format_layout.addRow("Bit Depth:", QLabel(f"{audio.bit_depth} Bit"))
        
        if audio.format_info:
            for key, value in audio.format_info.items():
                if key not in ("format", "note"):
                    format_layout.addRow(f"{key}:", QLabel(str(value)))
        
        layout.addWidget(format_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        stats_layout.addRow("Duration:", QLabel(format_time(audio.duration_seconds)))
        stats_layout.addRow("Samples:", QLabel(f"{audio.num_samples:,}"))
        stats_layout.addRow("Data Size:", QLabel(f"{audio.data.nbytes / 1024 / 1024:.2f} MB (in memory)"))
        layout.addWidget(stats_group)
        
        # Notes
        if audio.format_info.get("note"):
            note_label = QLabel(audio.format_info["note"])
            note_label.setStyleSheet("color: #888; font-style: italic; padding: 8px;")
            note_label.setWordWrap(True)
            layout.addWidget(note_label)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

