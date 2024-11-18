# pymobi/core/data.py

import mne

class PyMoBIData:
    """Data container for EEG and motion data."""
    
    def __init__(self, raw_mne: mne.io.Raw, motion_data=None):
        self.mne_raw = raw_mne  # MNE Raw object for EEG data.
        self.motion_data = motion_data  # Optional motion data.
        self.processing_history = []  # Track all processing steps.
    
    def add_processing_step(self, step_name: str, params: dict):
        """Record a processing step."""
        self.processing_history.append({"step": step_name, "params": params})
    
    def to_mne(self):
        """Return the MNE Raw object."""
        return self.mne_raw
    
    def sync_motion_events(self):
        """Synchronize motion events with EEG data."""
        if self.motion_data:
            # Implement synchronization logic here.
            pass