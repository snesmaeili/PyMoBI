# pymobi/viz/signals.py

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
        
    def plot_data_overview(self, 
                          data: PyMoBIData,
                          stage: str = "raw",
                          n_segments: int = 6,
                          segment_duration: float = 10.0) -> None:
        """
        Create overview plot of data at different time points.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
        stage : str
            Processing stage identifier
        n_segments : int
            Number of segments to plot
        segment_duration : float
            Duration of each segment in seconds
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{stage.capitalize()} Data Overview')
        
        # Create subplots for different time segments
        for i in range(n_segments):
            ax = plt.subplot(2, 3, i+1)
            
            # Calculate start time for this segment
            start_time = data.mne_raw.times[-1] / (n_segments + 1) * (i + 1)
            
            # Plot data segment
            self._plot_data_segment(data, ax, start_time, segment_duration)
            
            ax.set_title(f'Data section {i+1}: {start_time:.1f}s - {start_time + segment_duration:.1f}s')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channels')
            
        plt.tight_layout()
        
        # Save plot
        self._save_plot(data, fig, f"{stage}_data_overview")
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
            
        # Create figure for component properties
        fig = data.ica.plot_components(
            picks=range(min(n_components, len(data.ica.ch_names))),
            show=False
        )
        
        # Save plot
        self._save_plot(data, fig, "ica_components")
        plt.close()
        
    def plot_motion_events(self,
                          data: PyMoBIData,
                          event_window: float = 2.0) -> None:
        """
        Plot EEG data around motion events.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with motion events
        event_window : float
            Time window around events in seconds
        """
        if data.motion_events is None:
            raise ValueError("No motion events found in data")
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot EEG data around each event
        for event in data.motion_events:
            event_time = event['latency']
            start_idx = int((event_time - event_window) * data.mne_raw.info['sfreq'])
            end_idx = int((event_time + event_window) * data.mne_raw.info['sfreq'])
            
            if start_idx >= 0 and end_idx < data.mne_raw.n_times:
                times = np.arange(-event_window, event_window, 
                                1/data.mne_raw.info['sfreq'])
                ax.plot(times, 
                       data.mne_raw.get_data()[:, start_idx:end_idx].T,
                       alpha=0.5)
                
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title('EEG Data Around Motion Events')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        
        # Save plot
        self._save_plot(data, fig, "motion_events")
        plt.close()
        
    def _plot_data_segment(self,
                          data: PyMoBIData,
                          ax: plt.Axes,
                          start_time: float,
                          duration: float) -> None:
        """Plot a segment of EEG data."""
        start_idx = int(start_time * data.mne_raw.info['sfreq'])
        n_samples = int(duration * data.mne_raw.info['sfreq'])
        
        # Get data segment
        data_segment = data.mne_raw.get_data(
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