# pymobi/preprocessing/basic.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class BasicPreprocessing:
    """Basic preprocessing implementation for mobile EEG data."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize basic preprocessing with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing preprocessing parameters
        """
        self.config = config
        
    def run(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run all basic preprocessing steps.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with MNE Raw object
            
        Returns
        -------
        PyMoBIData
            Processed data container
        """
        # Store original data for quality check plots
        self._plot_raw_data(data, "raw")
        
        # Run preprocessing steps
        data = self._remove_channels(data)
        data = self._rename_channels(data)
        data = self._resample_data(data)
        data = self._add_channel_locations(data)
        data = self._add_reference_channel(data)
        data = self._set_channel_types(data)
        
        # Run Zapline+ if configured
        if hasattr(self.config, 'zapline_config') and self.config.zapline_config:
            data = self._run_zapline(data)
        
        # Generate and save quality check plots
        self._plot_raw_data(data, "preprocessed")
        
        return data
        
    def _remove_channels(self, data: PyMoBIData) -> PyMoBIData:
        """Remove specified channels."""
        if self.config.channels_to_remove:
            data.remove_channels(self.config.channels_to_remove)
            data.add_processing_step('remove_channels', {
                'removed_channels': self.config.channels_to_remove
            })
        return data
        
    def _rename_channels(self, data: PyMoBIData) -> PyMoBIData:
        """Rename channels according to configuration."""
        if self.config.rename_channels:
            data.mne_raw.rename_channels(self.config.rename_channels)
            data.add_processing_step('rename_channels', {
                'channel_mapping': self.config.rename_channels
            })
        return data
        
    def _resample_data(self, data: PyMoBIData) -> PyMoBIData:
        """Resample data to target frequency."""
        if self.config.resample_freq and data.mne_raw.info['sfreq'] != self.config.resample_freq:
            original_freq = data.mne_raw.info['sfreq']
            data.resample(self.config.resample_freq)
            data.add_processing_step('resample', {
                'original_freq': original_freq,
                'new_freq': self.config.resample_freq
            })
        return data
        
    def _add_channel_locations(self, data: PyMoBIData) -> PyMoBIData:
        """Add channel locations."""
        if self.config.channel_locations_filename:
            montage = self._load_channel_montage()
            data.mne_raw.set_montage(montage)
            data.add_processing_step('add_channel_locations', {
                'montage_file': self.config.channel_locations_filename
            })
        return data
        
    def _add_reference_channel(self, data: PyMoBIData) -> PyMoBIData:
        """Add reference channel if specified."""
        if self.config.ref_channel:
            data.add_reference_channel(self.config.ref_channel)
            data.add_processing_step('add_reference', {
                'reference': self.config.ref_channel
            })
        return data
        
    def _set_channel_types(self, data: PyMoBIData) -> PyMoBIData:
        """Set channel types (EEG, EOG, etc.)."""
        if self.config.eog_channels:
            ch_types = {ch: 'eog' for ch in self.config.eog_channels}
            data.mne_raw.set_channel_types(ch_types)
            data.add_processing_step('set_channel_types', {
                'eog_channels': self.config.eog_channels
            })
        return data
        
    def _run_zapline(self, data: PyMoBIData) -> PyMoBIData:
        """Run Zapline+ for line noise removal."""
        from .zapline import PyZaplinePlus
        
        zapline = PyZaplinePlus(self.config.zapline_config)
        return zapline.run(data)
        
    def _plot_raw_data(self, data: PyMoBIData, stage: str):
        """Generate and save raw data plots."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create 6 subplots showing different time segments
        for i in range(6):
            ax = plt.subplot(2, 3, i+1)
            start_time = data.mne_raw.times[-1] / 7 * (i + 1)
            duration = 10.0  # 10 second segments
            
            # Plot data segment
            data.mne_raw.plot(
                start=start_time,
                duration=duration,
                n_channels=min(20, len(data.mne_raw.ch_names)),
                scalings='auto',
                ax=ax,
                show=False
            )
            
            ax.set_title(f'Data section {i+1}')
            
        plt.tight_layout()
        
        # Save plot
        if hasattr(data, 'subject_id'):
            output_path = self._get_output_path(data)
            plt.savefig(output_path / f'sub-{data.subject_id}_{stage}_data.png')
        
        plt.close()
        
    def _get_output_path(self, data: PyMoBIData) -> Path:
        """Get output path for plots."""
        base_path = Path(self.config.study_folder) / self.config.eeg_preprocessing_folder
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"