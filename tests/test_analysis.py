# tests/test_analysis.py

import pytest
import mne
import numpy as np
from pymobi import PyMoBIConfig, PyMoBIData
from pymobi.analysis import SpectralAnalysis

class TestSpectralAnalysis:
    """Test suite for spectral analysis functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PyMoBIConfig(
            study_folder="test_data",
            filename_prefix="sub-",
            resample_freq=250.0
        )
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known spectral properties."""
        # Create synthetic data with known frequency components
        sfreq = 250.0
        t = np.arange(0, 10, 1/sfreq)
        n_channels = 32
        
        # Generate data with known frequency components
        data = np.zeros((n_channels, len(t)))
        for ch in range(n_channels):
            # Add 10 Hz oscillation
            data[ch, :] = np.sin(2 * np.pi * 10 * t)
            # Add 40 Hz oscillation
            data[ch, :] += 0.5 * np.sin(2 * np.pi * 40 * t)
            # Add noise
            data[ch, :] += 0.1 * np.random.randn(len(t))
            
        # Create MNE Raw object
        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels
        )
        raw = mne.io.RawArray(data, info)
        
        return PyMoBIData(raw, subject_id=1)
        
    def test_psd_computation(self, config, sample_data):
        """Test power spectral density computation."""
        spectral = SpectralAnalysis(config)
        
        # Compute PSD
        freqs, psd = spectral.compute_psd(
            sample_data,
            fmin=1,
            fmax=50,
            method='welch'
        )
        
        # Test frequency range
        assert np.min(freqs) >= 1
        assert np.max(freqs) <= 50
        
        # Test for known frequency peaks (10 Hz and 40 Hz)
        peak_freqs = freqs[np.argmax(np.mean(psd, axis=0))]
        assert any(np.abs(peak_freqs - 10) < 1)  # Peak near 10 Hz
        assert any(np.abs(peak_freqs - 40) < 1)  # Peak near 40 Hz
        
    def test_motion_locked_tfr(self, config, sample_data):
        """Test motion-locked time-frequency analysis."""
        # Add motion events
        sample_data.motion_events = [
            {'timestamp': 2.0, 'type': 'step'},
            {'timestamp': 4.0, 'type': 'step'},
            {'timestamp': 6.0, 'type': 'step'}
        ]
        
        spectral = SpectralAnalysis(config)
        
        # Compute TFR
        tfr = spectral.compute_motion_locked_tfr(
            sample_data,
            event_id={'step': 1},
            tmin=-0.5,
            tmax=0.5,
            freqs=np.logspace(1, 35, 20)
        )
        
        # Test TFR properties
        assert isinstance(tfr, mne.time_frequency.AverageTFR)
        assert tfr.data.shape[0] == len(sample_data.mne_raw.ch_names)
        
    @pytest.mark.parametrize("method", ['welch', 'multitaper'])
    def test_psd_methods(self, config, sample_data, method):
        """Test different PSD computation methods."""
        spectral = SpectralAnalysis(config)
        
        freqs, psd = spectral.compute_psd(
            sample_data,
            fmin=1,
            fmax=50,
            method=method
        )
        
        # Test output shapes
        assert len(freqs) > 0
        assert psd.shape[0] == len(sample_data.mne_raw.ch_names)
        assert psd.shape[1] == len(freqs)
        
    def test_frequency_bands(self, config, sample_data):
        """Test frequency band analysis."""
        spectral = SpectralAnalysis(config)
        
        # Define frequency bands
        freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        # Compute and plot topographies
        spectral.plot_psd_topography(
            sample_data,
            freqs=np.linspace(1, 50, 100),
            psd=np.random.rand(32, 100),
            freq_bands=freq_bands
        )
        
        # Test output files
        output_path = Path(config.study_folder) / 'spectral_analysis' / f'sub-{sample_data.subject_id}'
        assert (output_path / 'psd_topography.png').exists()