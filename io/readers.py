# pymobi/io/readers.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class DataReader:
    """Data reader for various formats with BIDS support."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize data reader with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing file paths and parameters
        """
        self.config = config
        
    def read_subject_data(self, subject_id: int) -> PyMoBIData:
        """
        Read data for a specific subject.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier
            
        Returns
        -------
        PyMoBIData
            Data container with loaded data
        """
        # Check if BIDS format
        bids_path = self._get_bids_path(subject_id)
        if bids_path.exists():
            return self.read_bids(subject_id)
        
        # Try raw data path
        raw_path = self._get_raw_path(subject_id)
        if raw_path.exists():
            return self.read_raw(subject_id)
            
        raise FileNotFoundError(f"No data found for subject {subject_id}")
        
    def read_bids(self, subject_id: int) -> PyMoBIData:
        """
        Read BIDS-formatted data.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier
            
        Returns
        -------
        PyMoBIData
            Data container with loaded data
        """
        bids_path = self._get_bids_path(subject_id)
        
        # Read BIDS data using MNE-BIDS
        raw = mne.io.read_raw_bids(bids_path)
        
        # Read motion data if available
        motion_data = self._read_motion_data(subject_id)
        
        return PyMoBIData(raw, subject_id=subject_id, motion_data=motion_data)
        
    def read_raw(self, subject_id: int) -> PyMoBIData:
        """
        Read raw data files.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier
            
        Returns
        -------
        PyMoBIData
            Data container with loaded data
        """
        raw_path = self._get_raw_path(subject_id)
        
        # Determine file type and read accordingly
        if raw_path.suffix == '.fif':
            raw = mne.io.read_raw_fif(raw_path, preload=True)
        elif raw_path.suffix == '.vhdr':
            raw = mne.io.read_raw_brainvision(raw_path, preload=True)
        elif raw_path.suffix == '.xdf':
            raw = self._read_xdf(raw_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_path.suffix}")
            
        # Read motion data if available
        motion_data = self._read_motion_data(subject_id)
        
        return PyMoBIData(raw, subject_id=subject_id, motion_data=motion_data)
        
    def _read_xdf(self, file_path: Path) -> mne.io.Raw:
        """Read XDF file and convert to MNE Raw."""
        # Implementation of XDF reading
        # This would integrate with pylsl or other XDF readers
        pass
        
    def _read_motion_data(self, subject_id: int) -> Optional[Dict]:
        """Read motion data if available."""
        motion_path = self._get_motion_path(subject_id)
        if not motion_path.exists():
            return None
            
        # Read motion data
        # This would integrate with motion data readers (e.g., KielMAT)
        pass
        
    def _get_bids_path(self, subject_id: int) -> Path:
        """Get BIDS data path for subject."""
        return (Path(self.config.study_folder) / 
                self.config.bids_data_folder / 
                f"sub-{subject_id}")
        
    def _get_raw_path(self, subject_id: int) -> Path:
        """Get raw data path for subject."""
        return (Path(self.config.study_folder) / 
                self.config.raw_eeglab_data_folder / 
                f"sub-{subject_id}" / 
                self.config.merged_filename)
        
    def _get_motion_path(self, subject_id: int) -> Path:
        """Get motion data path for subject."""
        return (Path(self.config.study_folder) / 
                self.config.motion_analysis_folder / 
                f"sub-{subject_id}" / 
                self.config.merged_motion_filename)

class BIDSWriter:
    """Export data to BIDS format."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize BIDS writer with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing file paths and parameters
        """
        self.config = config
        
    def write_bids(self, data: PyMoBIData) -> None:
        """
        Write data in BIDS format.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container to write
        """
        if data.subject_id is None:
            raise ValueError("Subject ID must be set to write BIDS data")
            
        # Create BIDS path
        bids_path = self._get_bids_path(data.subject_id)
        bids_path.mkdir(parents=True, exist_ok=True)
        
        # Write EEG data
        self._write_eeg_bids(data, bids_path)
        
        # Write motion data if available
        if data.motion_data is not None:
            self._write_motion_bids(data, bids_path)
            
        # Write metadata
        self._write_metadata(data, bids_path)
        
    def _write_eeg_bids(self, data: PyMoBIData, bids_path: Path) -> None:
        """Write EEG data in BIDS format."""
        # Convert to BIDS format using MNE-BIDS
        mne.io.Raw.save(data.mne_raw, 
                       bids_path / 'eeg' / 'raw.fif',
                       overwrite=True)
        
    def _write_motion_bids(self, data: PyMoBIData, bids_path: Path) -> None:
        """Write motion data in BIDS format."""
        # Implementation of motion data writing in BIDS format
        pass
        
    def _write_metadata(self, data: PyMoBIData, bids_path: Path) -> None:
        """Write metadata in BIDS format."""
        metadata = {
            'TaskName': 'MoBI',
            'SamplingFrequency': data.mne_raw.info['sfreq'],
            'EEGReference': self.config.ref_channel,
            'EEGChannelCount': len(data.mne_raw.ch_names),
            'EOGChannelCount': len(self.config.eog_channels),
            'ProcessingSteps': data.processing_history
        }
        
        with open(bids_path / 'dataset_description.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _get_bids_path(self, subject_id: int) -> Path:
        """Get BIDS path for subject."""
        return (Path(self.config.study_folder) / 
                self.config.bids_data_folder / 
                f"sub-{subject_id}")