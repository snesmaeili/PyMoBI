# pymobi/core/config.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PyMoBIConfig:
    """Configuration management for mobile EEG processing."""
    mne_preprocessing: Dict[str, Any]
    motion_preprocessing: Dict[str, Any]
    custom_processing: Dict[str, Any]
    visualization: Dict[str, Any]