"""
Utility module for Audio Analyzer.

Contains helper functions used by both Core and GUI.
"""

from .formatting import (
    format_time,
    format_frequency,
    format_db,
    samples_to_time_str,
)

__all__ = [
    "format_time",
    "format_frequency", 
    "format_db",
    "samples_to_time_str",
]

