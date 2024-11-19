# pymobi/preprocessing/artifacts.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import asrpy
from pathlib import Path
import matplotlib.pyplot as plt
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class ArtifactRemoval:
    """Artifact removal implementation for mobile EEG data."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize artifact removal with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing artifact removal parameters
        """
        self.config = config
        
    def run(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run complete artifact removal pipeline.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with MNE Raw object
            
        Returns
        -------
        PyMoBIData
            Processed data container
        """
        # Store original data for comparison
        original_data = data.mne_raw.get_data().copy()
        
        # Run artifact removal steps
        data = self.detect_bad_channels(data)
        data = self.interpolate_channels(data)
        
        if hasattr(self.config, 'use_asr') and self.config.use_asr:
            data = self.run_asr(data)
            
        # Generate quality check plots
        self._plot_artifact_removal(original_data, data)
        
        return data
        
    def detect_bad_channels(self, data: PyMoBIData) -> PyMoBIData:
        """
        Detect bad channels using correlation analysis.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
            
        Returns
        -------
        PyMoBIData
            Data container with detected bad channels
        """
        # Get data array
        eeg_data = data.get_data()
        
        # Parameters
        n_iterations = self.config.chan_detect_num_iter
        correlation_threshold = self.config.chancorr_crit
        detection_threshold = self.config.chan_detected_fraction_threshold
        
        # Initialize detection arrays
        n_channels = len(data.mne_raw.ch_names)
        detection_count = np.zeros(n_channels)
        
        # Run multiple iterations
        for _ in range(n_iterations):
            # Calculate channel correlations
            correlations = np.corrcoef(eeg_data)
            
            # Detect bad channels
            bad_channels = np.where(np.mean(correlations, axis=1) < correlation_threshold)[0]
            detection_count[bad_channels] += 1
            
        # Get final bad channels
        final_bad_channels = np.where(detection_count / n_iterations > detection_threshold)[0]
        bad_channel_names = [data.mne_raw.ch_names[i] for i in final_bad_channels]
        
        # Store bad channels
        data.bad_channels = bad_channel_names
        
        # Add processing step
        data.add_processing_step('bad_channel_detection', {
            'n_iterations': n_iterations,
            'correlation_threshold': correlation_threshold,
            'detection_threshold': detection_threshold,
            'bad_channels': bad_channel_names
        })
        
        return data
        
    def interpolate_channels(self, data: PyMoBIData) -> PyMoBIData:
        """
        Interpolate bad channels.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with detected bad channels
            
        Returns
        -------
        PyMoBIData
            Data container with interpolated channels
        """
        if data.bad_channels:
            # Mark bad channels
            data.mne_raw.info['bads'] = data.bad_channels
            
            # Interpolate
            data.mne_raw.interpolate_bads(reset_bads=True)
            
            # Store interpolated channels
            data.interpolated_channels = data.bad_channels
            
            # Add processing step
            data.add_processing_step('channel_interpolation', {
                'interpolated_channels': data.bad_channels
            })
            
        return data
        
    def run_asr(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run ASR for artifact removal.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
            
        Returns
        -------
        PyMoBIData
            Data container with ASR applied
        """
        # Initialize ASR
        asr = asrpy.ASR(
            sfreq=data.mne_raw.info['sfreq'],
            cutoff=self.config.asr_cutoff
        )
        
        # Fit ASR
        asr.fit(data.mne_raw)
        
        # Transform data
        data.mne_raw = asr.transform(data.mne_raw)
        
        # Add processing step
        data.add_processing_step('asr', {
            'cutoff': self.config.asr_cutoff
        })
        
        return data
        
    def remove_ica_artifacts(self, data: PyMoBIData) -> PyMoBIData:
        """
        Remove artifacts using ICA and ICLabel.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with ICA solution
            
        Returns
        -------
        PyMoBIData
            Data container with artifacts removed
        """
        if data.ica is None:
            from .ica import ICAProcessor
            ica_proc = ICAProcessor(self.config)
            data = ica_proc.run(data)
            
        # Run ICLabel if not already done
        if data.iclabel_scores is None:
            data = self._run_iclabel(data)
            
        # Remove artifact components
        artifact_idx = self._get_artifact_components(data)
        data.ica.exclude = artifact_idx
        
        # Apply ICA
        data.mne_raw = data.ica.apply(data.mne_raw)
        
        # Add processing step
        data.add_processing_step('ica_artifact_removal', {
            'removed_components': artifact_idx.tolist()
        })
        
        return data
        
    def _run_iclabel(self, data: PyMoBIData) -> PyMoBIData:
        """Run ICLabel classification."""
        # Implementation of ICLabel
        pass
        
    def _get_artifact_components(self, data: PyMoBIData) -> np.ndarray:
        """Get indices of artifact components based on ICLabel scores."""
        # Implementation of artifact component selection
        pass
        
    def _plot_artifact_removal(self, original_data: np.ndarray, data: PyMoBIData):
        """Generate comparison plots of original and cleaned data."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle('Artifact Removal Results')
        
        # Plot parameters
        plot_length = int(10 * data.mne_raw.info['sfreq'])  # 10 seconds
        times = np.arange(plot_length) / data.mne_raw.info['sfreq']
        cleaned_data = data.mne_raw.get_data()
        
        # Original data
        axes[0].plot(times, original_data[:, :plot_length].T)
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        
        # Cleaned data
        axes[1].plot(times, cleaned_data[:, :plot_length].T)
        axes[1].set_title('Cleaned Data')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        
        # Difference
        axes[2].plot(times, (original_data[:, :plot_length] - cleaned_data[:, :plot_length]).T)
        axes[2].set_title('Removed Components')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        
        plt.tight_layout()
        
        # Save plot
        if hasattr(data, 'subject_id'):
            output_path = self._get_output_path(data)
            plt.savefig(output_path / f'sub-{data.subject_id}_artifact_removal.png')
        plt.close()