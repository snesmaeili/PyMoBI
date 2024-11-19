
import mne
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
import matplotlib.pyplot as plt

class PyMoBIData:
    """Data container for mobile EEG and motion data processing."""
    
    def __init__(self, 
                 raw_mne: mne.io.Raw, 
                 subject_id: Optional[int] = None,
                 motion_data: Optional[Dict] = None):
        """
        Initialize PyMoBI data container.
        
        Parameters
        ----------
        raw_mne : mne.io.Raw
            MNE Raw object containing EEG data
        subject_id : Optional[int]
            Subject identifier
        motion_data : Optional[Dict]
            Dictionary containing motion capture data
        """
        self.mne_raw = raw_mne
        self.subject_id = subject_id
        self.motion_data = motion_data
        
        # Processing history and metadata
        self.processing_history = []
        self.metadata = {}
        
        # Bad channels
        self.bad_channels = []
        self.interpolated_channels = []
        
        # ICA related attributes
        self.ica = None
        self.ica_components = None
        self.ica_excluded = []
        self.iclabel_scores = None
        
        # Motion events
        self.gait_events = None
        self.motion_events = None
        
        # Quality metrics
        self.quality_metrics = {}
        
    def add_processing_step(self, step_name: str, params: Dict[str, Any]) -> None:
        """
        Record a processing step with parameters.
        
        Parameters
        ----------
        step_name : str
            Name of the processing step
        params : Dict[str, Any]
            Parameters used in the processing step
        """
        self.processing_history.append({
            'step': step_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        
    def get_data(self) -> np.ndarray:
        """Get EEG data array."""
        return self.mne_raw.get_data()
        
    def get_times(self) -> np.ndarray:
        """Get time points."""
        return self.mne_raw.times
        
    def remove_channels(self, channels: List[str]) -> None:
        """
        Remove specified channels.
        
        Parameters
        ----------
        channels : List[str]
            List of channel names to remove
        """
        self.mne_raw.drop_channels(channels)
        self.add_processing_step('remove_channels', {'channels': channels})
        
    def resample(self, sfreq: float) -> None:
        """
        Resample data to new frequency.
        
        Parameters
        ----------
        sfreq : float
            New sampling frequency
        """
        self.mne_raw.resample(sfreq)
        self.add_processing_step('resample', {'new_sfreq': sfreq})
        
    def add_reference_channel(self, ref_channel: str) -> None:
        """
        Add reference channel.
        
        Parameters
        ----------
        ref_channel : str
            Name of reference channel to add
        """
        self.mne_raw.add_reference_channels(ref_channel)
        self.add_processing_step('add_reference', {'channel': ref_channel})
        
    def set_channel_types(self, channel_types: Dict[str, str]) -> None:
        """
        Set channel types (EEG, EOG, etc.).
        
        Parameters
        ----------
        channel_types : Dict[str, str]
            Dictionary mapping channel names to types
        """
        self.mne_raw.set_channel_types(channel_types)
        self.add_processing_step('set_channel_types', {'types': channel_types})
        
    def apply_average_reference(self) -> None:
        """Apply average reference."""
        self.mne_raw.set_eeg_reference('average', projection=True)
        self.add_processing_step('average_reference', {})
        
    def add_motion_events(self, events: Dict[str, Any]) -> None:
        """
        Add motion events to the data.
        
        Parameters
        ----------
        events : Dict[str, Any]
            Dictionary containing motion events
        """
        self.motion_events = events
        self.add_processing_step('add_motion_events', {
            'n_events': len(events) if events else 0
        })
        
    def save(self, 
             filename: Union[str, Path], 
             overwrite: bool = False) -> None:
        """
        Save processed data and metadata.
        
        Parameters
        ----------
        filename : Union[str, Path]
            Path to save the data
        overwrite : bool
            Whether to overwrite existing files
        """
        filename = Path(filename)
        
        # Save MNE Raw object
        self.mne_raw.save(filename, overwrite=overwrite)
        
        # Prepare metadata
        metadata = {
            'subject_id': self.subject_id,
            'processing_history': self.processing_history,
            'bad_channels': self.bad_channels,
            'interpolated_channels': self.interpolated_channels,
            'ica_excluded': self.ica_excluded,
            'quality_metrics': self.quality_metrics,
            'metadata': self.metadata
        }
        
        # Save metadata
        metadata_file = filename.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    @classmethod
    def load(cls, 
             filename: Union[str, Path], 
             preload: bool = True) -> 'PyMoBIData':
        """
        Load saved data and metadata.
        
        Parameters
        ----------
        filename : Union[str, Path]
            Path to the data file
        preload : bool
            Whether to preload the data into memory
            
        Returns
        -------
        PyMoBIData
            Loaded data container
        """
        filename = Path(filename)
        
        # Load MNE Raw object
        raw = mne.io.read_raw(filename, preload=preload)
        
        # Load metadata if exists
        metadata_file = filename.with_suffix('.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Create instance
            instance = cls(raw, subject_id=metadata.get('subject_id'))
            
            # Restore attributes
            instance.processing_history = metadata['processing_history']
            instance.bad_channels = metadata['bad_channels']
            instance.interpolated_channels = metadata['interpolated_channels']
            instance.ica_excluded = metadata['ica_excluded']
            instance.quality_metrics = metadata['quality_metrics']
            instance.metadata = metadata['metadata']
            
            return instance
        else:
            return cls(raw)
            
    def plot_data_overview(self, 
                          time_window: float = 10.0,
                          n_channels: int = 20) -> plt.Figure:
        """
        Plot EEG data overview.
        
        Parameters
        ----------
        time_window : float
            Time window to display in seconds
        n_channels : int
            Number of channels to display
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        fig = plt.figure(figsize=(15, 8))
        self.mne_raw.plot(duration=time_window, n_channels=n_channels)
        return fig