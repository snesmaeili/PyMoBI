# pymobi/motion/gait.py

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class MotionProcessor:
    """Motion data processing and gait analysis integration."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize motion processor with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing motion processing parameters
        """
        self.config = config
        
    def run(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run motion processing pipeline.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with motion data
            
        Returns
        -------
        PyMoBIData
            Data container with processed motion data
        """
        if data.motion_data is None:
            raise ValueError("No motion data available")
            
        # Store original data for comparison
        original_motion = data.motion_data.copy()
        
        try:
            # Process motion data
            data = self._preprocess_motion(data)
            data = self._detect_gait_events(data)
            data = self._sync_with_eeg(data)
            
            # Generate quality check plots
            self._plot_motion_processing(data, original_motion)
            
        except Exception as e:
            print(f"Error during motion processing: {str(e)}")
            raise
            
        return data
        
    def _preprocess_motion(self, data: PyMoBIData) -> PyMoBIData:
        """Preprocess motion data."""
        # Apply lowpass filter
        data.motion_data = self._apply_lowpass_filter(
            data.motion_data,
            self.config.lowpass_motion
        )
        
        # Calculate derivatives if needed
        if self.config.compute_derivatives:
            data.motion_data = self._compute_derivatives(data.motion_data)
            
            # Apply lowpass to derivatives
            data.motion_data = self._apply_lowpass_filter(
                data.motion_data,
                self.config.lowpass_motion_after_derivative
            )
            
        data.add_processing_step('motion_preprocessing', {
            'lowpass_freq': self.config.lowpass_motion,
            'compute_derivatives': self.config.compute_derivatives
        })
        
        return data
        
    def _detect_gait_events(self, data: PyMoBIData) -> PyMoBIData:
        """
        Detect gait events using KielMAT integration.
        
        This would integrate with KielMAT's gait detection functionality:
        https://github.com/neurogeriatricskiel/KielMAT
        """
        try:
            # Import KielMAT (assuming it's installed)
            import kielmat
            
            # Detect gait events
            events = kielmat.detect_gait_events(
                data.motion_data,
                fs=data.mne_raw.info['sfreq']
            )
            
            # Store events
            data.gait_events = events
            
            data.add_processing_step('gait_detection', {
                'n_events': len(events),
                'method': 'kielmat'
            })
            
        except ImportError:
            print("KielMAT not found. Please install it for gait analysis.")
            raise
            
        return data
        
    def _sync_with_eeg(self, data: PyMoBIData) -> PyMoBIData:
        """Synchronize motion events with EEG data."""
        if data.gait_events is None:
            return data
            
        # Convert motion events to MNE annotations
        annotations = []
        for event in data.gait_events:
            annotations.append({
                'onset': event['timestamp'],
                'duration': 0,
                'description': event['type']
            })
            
        # Add annotations to EEG data
        data.mne_raw.annotations.append(
            onset=[ann['onset'] for ann in annotations],
            duration=[ann['duration'] for ann in annotations],
            description=[ann['description'] for ann in annotations]
        )
        
        data.add_processing_step('motion_sync', {
            'n_events_synced': len(annotations)
        })
        
        return data
        
    def _apply_lowpass_filter(self, 
                            motion_data: Dict[str, np.ndarray],
                            cutoff_freq: float) -> Dict[str, np.ndarray]:
        """Apply lowpass filter to motion data."""
        from scipy.signal import butter, filtfilt
        
        # Design filter
        nyq = 0.5 * self.config.resample_freq
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(4, normal_cutoff, btype='low')
        
        # Apply filter to each motion channel
        filtered_data = {}
        for channel, data in motion_data.items():
            filtered_data[channel] = filtfilt(b, a, data)
            
        return filtered_data
        
    def _compute_derivatives(self, 
                           motion_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute motion derivatives."""
        derivatives = {}
        for channel, data in motion_data.items():
            # First derivative (velocity)
            derivatives[f"{channel}_vel"] = np.gradient(data)
            
            # Second derivative (acceleration)
            derivatives[f"{channel}_acc"] = np.gradient(derivatives[f"{channel}_vel"])
            
        # Combine original and derivatives
        return {**motion_data, **derivatives}
        
    def _plot_motion_processing(self, 
                              data: PyMoBIData,
                              original_motion: Dict[str, np.ndarray]):
        """Generate motion processing plots."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle('Motion Processing Results')
        
        # Select a representative channel
        channel = list(original_motion.keys())[0]
        times = np.arange(len(original_motion[channel])) / data.mne_raw.info['sfreq']
        
        # Original motion
        axes[0].plot(times, original_motion[channel])
        axes[0].set_title('Original Motion')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        
        # Processed motion
        axes[1].plot(times, data.motion_data[channel])
        axes[1].set_title('Processed Motion')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        
        # Gait events
        if data.gait_events is not None:
            axes[2].plot(times, data.motion_data[channel])
            event_times = [event['timestamp'] for event in data.gait_events]
            event_types = [event['type'] for event in data.gait_events]
            
            for t, type_ in zip(event_times, event_types):
                axes[2].axvline(t, color='r', linestyle='--', alpha=0.5)
                
        axes[2].set_title('Detected Gait Events')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        
        plt.tight_layout()
        
        # Save plot
        if hasattr(data, 'subject_id'):
            output_path = self._get_output_path(data)
            plt.savefig(output_path / f'sub-{data.subject_id}_motion_processing.png')
        plt.close()
        
    def _get_output_path(self, data: PyMoBIData) -> Path:
        """Get output path for plots."""
        base_path = Path(self.config.study_folder) / self.config.motion_analysis_folder
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"