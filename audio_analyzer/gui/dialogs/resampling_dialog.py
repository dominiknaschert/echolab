"""
Resampling und Downmix Dialog

Erlaubt dem Nutzer die bewusste Entscheidung über Signalkonvertierung.
Keine automatischen Manipulationen!
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QRadioButton, QButtonGroup,
    QComboBox, QPushButton, QCheckBox, QSpinBox,
)
from PySide6.QtCore import Qt

from ...core.audio_io import AudioFile
from ...utils.formatting import format_sample_rate, format_channels


class ResamplingDialog(QDialog):
    """
    Dialog for explicit resampling/downmix decisions.
    
    Returns the user's choices, but does NOT modify the audio.
    The calling code must apply the changes.
    """
    
    def __init__(self, audio: AudioFile, parent=None):
        super().__init__(parent)
        self.audio = audio
        self.setWindowTitle("Signalkonvertierung")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Current info
        info_label = QLabel(
            f"Aktuelle Datei: {format_sample_rate(audio.sample_rate)}, "
            f"{format_channels(audio.channels)}"
        )
        info_label.setStyleSheet("font-weight: bold; padding: 8px;")
        layout.addWidget(info_label)
        
        # Warning
        warning = QLabel(
            "⚠ Signalkonvertierung verändert die Audiodaten.\n"
            "Für reproduzierbare Analyse sollten Originaldaten verwendet werden."
        )
        warning.setStyleSheet("color: #f0ad4e; padding: 8px;")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        
        # Channel options (only for stereo)
        if audio.channels == 2:
            channel_group = QGroupBox("Kanalbehandlung")
            channel_layout = QVBoxLayout(channel_group)
            
            self.channel_button_group = QButtonGroup(self)
            
            self.rb_keep_stereo = QRadioButton("Stereo beibehalten")
            self.rb_keep_stereo.setChecked(True)
            self.channel_button_group.addButton(self.rb_keep_stereo, 0)
            channel_layout.addWidget(self.rb_keep_stereo)
            
            self.rb_left = QRadioButton("Nur linker Kanal")
            self.channel_button_group.addButton(self.rb_left, 1)
            channel_layout.addWidget(self.rb_left)
            
            self.rb_right = QRadioButton("Nur rechter Kanal")
            self.channel_button_group.addButton(self.rb_right, 2)
            channel_layout.addWidget(self.rb_right)
            
            self.rb_downmix = QRadioButton("Downmix zu Mono: (L+R)/2")
            self.channel_button_group.addButton(self.rb_downmix, 3)
            channel_layout.addWidget(self.rb_downmix)
            
            note = QLabel(
                "Hinweis: Downmix erfolgt ohne Energiekompensation.\n"
                "Bei korreliertem Material kann dies zu Pegelreduktion führen."
            )
            note.setStyleSheet("color: #888; font-size: 11px;")
            channel_layout.addWidget(note)
            
            layout.addWidget(channel_group)
        else:
            self.channel_button_group = None
        
        # Resampling options
        resample_group = QGroupBox("Resampling")
        resample_layout = QVBoxLayout(resample_group)
        
        self.cb_resample = QCheckBox("Samplerate ändern")
        self.cb_resample.stateChanged.connect(self._on_resample_toggled)
        resample_layout.addWidget(self.cb_resample)
        
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Neue Samplerate:"))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["22050", "44100", "48000", "88200", "96000"])
        self.sr_combo.setCurrentText("44100")
        self.sr_combo.setEnabled(False)
        sr_layout.addWidget(self.sr_combo)
        sr_layout.addWidget(QLabel("Hz"))
        sr_layout.addStretch()
        resample_layout.addLayout(sr_layout)
        
        resample_note = QLabel(
            "Resampling verwendet Polyphasen-Algorithmus mit Anti-Aliasing."
        )
        resample_note.setStyleSheet("color: #888; font-size: 11px;")
        resample_layout.addWidget(resample_note)
        
        layout.addWidget(resample_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Anwenden")
        apply_btn.clicked.connect(self.accept)
        btn_layout.addWidget(apply_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_resample_toggled(self, state):
        """Handle resample checkbox toggle."""
        self.sr_combo.setEnabled(state == Qt.CheckState.Checked.value)
    
    def get_channel_option(self) -> str:
        """
        Get selected channel option.
        
        Returns:
            "stereo", "left", "right", or "downmix"
        """
        if self.channel_button_group is None:
            return "mono"  # Already mono
        
        selected = self.channel_button_group.checkedId()
        return {0: "stereo", 1: "left", 2: "right", 3: "downmix"}.get(selected, "stereo")
    
    def get_target_sample_rate(self) -> int | None:
        """
        Get target sample rate if resampling is requested.
        
        Returns:
            Target sample rate or None if no resampling
        """
        if self.cb_resample.isChecked():
            return int(self.sr_combo.currentText())
        return None

