# pymobi/analysis/spectral.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class SpectralAnalysis:
    """Spectral analysis tools for mobile EEG data."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize spectral analysis with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing spectral analysis parameters
        """
        self.config = config
        
    def compute_psd(self, 
                    data: PyMoBIData,
                    fmin: float = 1.0,
                    fmax: float = 45.0,
                    tmin: Optional[float] = None,
                    tmax: Optional[float] = None,
                    method: str = 'welch',
                    n_fft: Optional[int] = None,
                    n_overlap: Optional[int] = None,
                    window: str = 'hamming') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
        fmin : float
            Minimum frequency
        fmax : float
            Maximum frequency
        tmin : Optional[float]
            Start time for analysis
        tmax : Optional[float]
            End time for analysis
        method : str
            Method for PSD computation ('welch' or 'multitaper')
        n_fft : Optional[int]
            Number of FFT points
        n_overlap : Optional[int]
            Number of overlap points
        window : str
            Window function
            
        Returns
        -------
        freqs : np.ndarray
            Frequency points
        psd : np.ndarray
            PSD values
        """
        # Set default parameters if not specified
        if n_fft is None:
            n_fft = int(4 * data.mne_raw.info['sfreq'])
            
        if n_overlap is None:
            n_overlap = n_fft // 2
            
        # Compute PSD
        if method == 'welch':
            psds, freqs = mne.time_frequency.psd_welch(
                data.mne_raw,
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                n_fft=n_fft,
                n_overlap=n_overlap,
                window=window
            )
        elif method == 'multitaper':
            psds, freqs = mne.time_frequency.psd_multitaper(
                data.mne_raw,
                fmin=fmin,
                fmax=fmax,
                tmin=tmin,
                tmax=tmax,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return freqs, psds
        
    def compute_motion_locked_tfr(self,
                                data: PyMoBIData,
                                event_id: Dict[str, int],
                                tmin: float = -1.0,
                                tmax: float = 1.0,
                                freqs: np.ndarray = np.logspace(1, 35, 20),
                                n_cycles: Union[int, List[int]] = 7,
                                baseline: Optional[Tuple[float, float]] = None) -> mne.time_frequency.AverageTFR:
        """
        Compute time-frequency representation locked to motion events.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with motion events
        event_id : Dict[str, int]
            Event IDs to analyze
        tmin : float
            Start time relative to events
        tmax : float
            End time relative to events
        freqs : np.ndarray
            Frequencies to analyze
        n_cycles : Union[int, List[int]]
            Number of cycles for wavelets
        baseline : Optional[Tuple[float, float]]
            Baseline period for normalization
            
        Returns
        -------
        tfr : mne.time_frequency.AverageTFR
            Time-frequency representation
        """
        if data.motion_events is None:
            raise ValueError("No motion events found in data")
            
        # Create epochs around motion events
        epochs = mne.Epochs(
            data.mne_raw,
            data.motion_events,
            event_id,
            tmin,
            tmax,
            baseline=baseline,
            preload=True
        )
        
        # Compute TFR
        tfr = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True
        )
        
        return tfr
        
    def plot_psd_topography(self,
                           data: PyMoBIData,
                           freqs: np.ndarray,
                           psd: np.ndarray,
                           freq_bands: Dict[str, Tuple[float, float]]) -> None:
        """
        Plot PSD topography for different frequency bands.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
        freqs : np.ndarray
            Frequency points
        psd : np.ndarray
            PSD values
        freq_bands : Dict[str, Tuple[float, float]]
            Frequency bands to plot
        """
        n_bands = len(freq_bands)
        fig, axes = plt.subplots(1, n_bands, figsize=(5*n_bands, 4))
        
        if n_bands == 1:
            axes = [axes]
            
        for ax, (band_name, (fmin, fmax)) in zip(axes, freq_bands.items()):
            # Find frequency indices
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psd[:, freq_mask], axis=1)
            
            # Create topomap
            mne.viz.plot_topomap(
                band_power,
                data.mne_raw.info,
                axes=ax,
                show=False
            )
            
            ax.set_title(f'{band_name} ({fmin}-{fmax} Hz)')
            
        plt.tight_layout()
        
        # Save plot
        self._save_plot(data, fig, "psd_topography")
        plt.close()
        
    def plot_tfr_comparison(self,
                           data: PyMoBIData,
                           tfr: mne.time_frequency.AverageTFR,
                           channels: Optional[List[str]] = None) -> None:
        """
        Plot time-frequency comparison for selected channels.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container
        tfr : mne.time_frequency.AverageTFR
            Time-frequency representation
        channels : Optional[List[str]]
            Channels to plot
        """
        if channels is None:
            channels = data.mne_raw.ch_names[:5]  # Plot first 5 channels by default
            
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 4*len(channels)))
        
        if len(channels) == 1:
            axes = [axes]
            
        for ax, channel in zip(axes, channels):
            tfr.plot(picks=[channel], axes=ax, show=False)
            ax.set_title(f'Channel {channel}')
            
        plt.tight_layout()
        
        # Save plot
        self._save_plot(data, fig, "tfr_comparison")
        plt.close()
        
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
        base_path = Path(self.config.study_folder) / 'spectral_analysis'
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"