# pymobi/viz/plots.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
from pathlib import Path
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class SignalVisualizer:
    """Enhanced visualization tools for mobile EEG data."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize visualizer with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing visualization parameters
        """
        self.config = config
        
    def plot_raw_overview(self, 
                         data: PyMoBIData,
                         n_segments: int = 6,
                         segment_duration: float = 10.0,
                         filter_freq: float = 0.5) -> None:
        """
        Create overview plot of raw data at different time points.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
        n_segments : int
            Number of segments to plot
        segment_duration : float
            Duration of each segment in seconds
        filter_freq : float
            High-pass filter frequency for visualization
        """
        # Create temporary filtered data for plotting
        filtered_data = data.mne_raw.copy()
        filtered_data.filter(l_freq=filter_freq, h_freq=None)
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Raw Data Overview')
        
        # Create subplots for different time segments
        for i in range(n_segments):
            ax = plt.subplot(2, 3, i+1)
            
            # Calculate start time for this segment
            start_time = filtered_data.times[-1] / (n_segments + 1) * (i + 1)
            
            # Plot data segment
            self._plot_data_segment(filtered_data, ax, start_time, segment_duration)
            
            ax.set_title(f'Section {i+1}: {start_time:.1f}s - {start_time + segment_duration:.1f}s')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channels')
            
        plt.tight_layout()
        
        # Save plot
        self._save_plot(data, fig, "raw_overview")
        plt.close()
        
    def plot_preprocessing_comparison(self,
                                   data: PyMoBIData,
                                   original_data: np.ndarray,
                                   stage: str) -> None:
        """
        Plot comparison between original and processed data.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with processed data
        original_data : np.ndarray
            Original data array
        stage : str
            Processing stage identifier
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'{stage.capitalize()} Processing Results')
        
        # Plot parameters
        plot_length = int(10 * data.mne_raw.info['sfreq'])  # 10 seconds
        times = np.arange(plot_length) / data.mne_raw.info['sfreq']
        processed_data = data.mne_raw.get_data()
        
        # Original data
        axes[0].plot(times, original_data[:, :plot_length].T)
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        
        # Processed data
        axes[1].plot(times, processed_data[:, :plot_length].T)
        axes[1].set_title('Processed Data')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        
        # Difference
        axes[2].plot(times, (original_data[:, :plot_length] - 
                           processed_data[:, :plot_length]).T)
        axes[2].set_title('Difference')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        
        plt.tight_layout()
        
        # Save plot
        self._save_plot(data, fig, f"{stage}_comparison")
        plt.close()
        
    def plot_bad_channels(self,
                         data: PyMoBIData,
                         bad_channels: List[str],
                         detection_scores: np.ndarray) -> None:
        """
        Plot bad channel detection results.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
        bad_channels : List[str]
            List of detected bad channels
        detection_scores : np.ndarray
            Detection scores for each channel
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Bad Channel Detection Results')
        
        # Plot channel correlations
        ax1.bar(range(len(data.mne_raw.ch_names)), detection_scores)
        ax1.axhline(y=self.config.chancorr_crit, color='r', linestyle='--')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Detection Score')
        ax1.set_title('Channel Detection Scores')
        
        # Highlight bad channels
        bad_indices = [data.mne_raw.ch_names.index(ch) for ch in bad_channels]
        ax1.bar(bad_indices, detection_scores[bad_indices], color='r')
        
        # Plot topography
        mne.viz.plot_topomap(detection_scores, 
                            data.mne_raw.info, 
                            axes=ax2,
                            show=False)
        ax2.set_title('Topographic Distribution')
        
        plt.tight_layout()
        
        # Save plot
        self._save_plot(data, fig, "bad_channels")
        plt.close()
        
    def plot_ica_components(self,
                          data: PyMoBIData,
                          n_components: Optional[int] = 20) -> None:
        """
        Plot ICA components.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with ICA solution
        n_components : Optional[int]
            Number of components to plot
        """
        if data.ica is None:
            raise ValueError("No ICA solution found in data")
            
        # Plot components
        fig = data.ica.plot_components(
            picks=range(min(n_components, len(data.ica.ch_names))),
            show=False
        )
        
        # Save plot
        self._save_plot(data, fig, "ica_components")
        plt.close()
        
    def _plot_data_segment(self,
                          raw: mne.io.Raw,
                          ax: plt.Axes,
                          start_time: float,
                          duration: float) -> None:
        """Plot a segment of EEG data."""
        start_idx = int(start_time * raw.info['sfreq'])
        n_samples = int(duration * raw.info['sfreq'])
        
        # Get data segment
        data_segment = raw.get_data(
            start=start_idx,
            stop=start_idx + n_samples
        )
        
        # Plot channels
        times = np.linspace(start_time, start_time + duration, data_segment.shape[1])
        ax.plot(times, data_segment.T + np.arange(data_segment.shape[0])[:, None] * 100)
        
    def _save_plot(self,
                   data: PyMoBIData,
                   fig: plt.Figure,
                   name: str) -> None:
        """Save plot to appropriate directory."""
        if hasattr(data, 'subject_id'):
            output_path = self._get_output_path(data)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as PNG
            fig.savefig(output_path / f'sub-{data.subject_id}_{name}.png')
            
            # Optionally save as vector graphics
            if hasattr(self.config, 'save_vector_graphics') and self.config.save_vector_graphics:
                fig.savefig(output_path / f'sub-{data.subject_id}_{name}.svg')
                
    def _get_output_path(self, data: PyMoBIData) -> Path:
        """Get output path for plots."""
        base_path = Path(self.config.study_folder) / 'figures'
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"