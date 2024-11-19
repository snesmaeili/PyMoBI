# pymobi/preprocessing/pyasr.py

import mne
import numpy as np
from typing import Optional, Dict, Any, Tuple
import asrpy
from pathlib import Path
import matplotlib.pyplot as plt
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class PyASR:
    """Python implementation of Artifact Subspace Reconstruction using ASRpy."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize PyASR with configuration parameters.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing ASR parameters:
            - asr_cutoff: float (default=20)
                Standard deviation cutoff for rejection
            - asr_window_len: float (default=0.5)
                Sliding window length in seconds
            - asr_max_bad_chans: float (default=0.1)
                Maximum fraction of bad channels per window
            - asr_use_clean_window: bool (default=True)
                Whether to use clean_windows for calibration
        """
        self.config = config
        
        # ASR parameters
        self.cutoff = getattr(config, 'asr_cutoff', 20)
        self.window_len = getattr(config, 'asr_window_len', 0.5)
        self.max_bad_chans = getattr(config, 'asr_max_bad_chans', 0.1)
        self.use_clean_window = getattr(config, 'asr_use_clean_window', True)
        
        # Initialize ASR object
        self.asr = None
        self.M = None  # State matrix for manual processing
        self.T = None  # Threshold matrix for manual processing
        
    def run(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run ASR on the EEG data.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with MNE Raw object
            
        Returns
        -------
        PyMoBIData
            Data container with cleaned data
        """
        # Store original data for comparison
        original_data = data.mne_raw.get_data().copy()
        
        try:
            if self.config.asr_use_mne:
                # MNE-based approach using ASRpy's high-level interface
                self._run_mne_based(data)
            else:
                # Manual approach using ASRpy's low-level functions
                self._run_manual(data)
            
            # Record processing step
            data.add_processing_step('asr', {
                'cutoff': self.cutoff,
                'window_len': self.window_len,
                'max_bad_chans': self.max_bad_chans,
                'removed_variance': self._calculate_removed_variance(
                    original_data, 
                    data.mne_raw.get_data()
                )
            })
            
            # Generate quality check plots
            self._plot_cleaning_comparison(original_data, data)
            
        except Exception as e:
            print(f"Error during ASR processing: {str(e)}")
            raise
            
        return data
        
    def _run_mne_based(self, data: PyMoBIData):
        """Run ASR using MNE-based interface."""
        # Initialize ASR
        self.asr = asrpy.ASR(
            sfreq=data.mne_raw.info['sfreq'],
            cutoff=self.cutoff
        )
        
        # Fit ASR
        self.asr.fit(data.mne_raw)
        
        # Transform data
        data.mne_raw = self.asr.transform(data.mne_raw)
        
    def _run_manual(self, data: PyMoBIData):
        """Run ASR using manual numpy array processing."""
        # Get data array and sampling frequency
        eeg_array = data.mne_raw.get_data()
        sfreq = data.mne_raw.info['sfreq']
        
        if self.use_clean_window:
            # Pre-clean windows for calibration
            pre_cleaned, _ = asrpy.clean_windows(
                eeg_array, 
                sfreq, 
                max_bad_chans=self.max_bad_chans
            )
        else:
            pre_cleaned = eeg_array
            
        # Calibrate ASR
        self.M, self.T = asrpy.asr_calibrate(
            pre_cleaned, 
            sfreq, 
            cutoff=self.cutoff
        )
        
        # Process data
        cleaned_array = asrpy.asr_process(
            eeg_array, 
            sfreq, 
            self.M, 
            self.T
        )
        
        # Update MNE Raw object
        data.mne_raw._data = cleaned_array
        
    def _calculate_removed_variance(self, 
                                  original_data: np.ndarray, 
                                  cleaned_data: np.ndarray) -> float:
        """Calculate the percentage of variance removed by ASR."""
        original_var = np.var(original_data)
        cleaned_var = np.var(cleaned_data)
        return (original_var - cleaned_var) / original_var * 100
        
    def _plot_cleaning_comparison(self, 
                                original_data: np.ndarray, 
                                data: PyMoBIData):
        """Generate comparison plots of original and cleaned data."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle('ASR Cleaning Results')
        
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
        axes[1].set_title('ASR Cleaned Data')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        
        # Difference
        axes[2].plot(times, (original_data[:, :plot_length] - 
                           cleaned_data[:, :plot_length]).T)
        axes[2].set_title('Removed Components')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        
        plt.tight_layout()
        
        # Save plot
        if hasattr(data, 'subject_id'):
            output_path = self._get_output_path(data)
            plt.savefig(output_path / f'sub-{data.subject_id}_asr_cleaning.png')
        plt.close()
        
    def _get_output_path(self, data: PyMoBIData) -> Path:
        """Get output path for plots."""
        base_path = Path(self.config.study_folder) / self.config.preprocessing_dir
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"