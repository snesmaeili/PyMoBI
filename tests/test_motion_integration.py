# tests/test_motion_integration.py

import pytest
import mne
import numpy as np
from pathlib import Path
from pymobi import PyMoBIConfig, PyMoBIData
from pymobi.motion import MotionProcessor

class TestMotionIntegration:
    """Test suite for motion data integration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PyMoBIConfig(
            study_folder=Path("test_data"),
            lowpass_motion=8.0,
            lowpass_motion_after_derivative=24.0,
            compute_derivatives=True
        )
    
    @pytest.fixture
    def sample_motion_data(self):
        """Create sample motion data."""
        # Create synthetic motion data
        time = np.arange(0, 10, 0.01)  # 10 seconds at 100 Hz
        motion_data = {
            'AccX': np.sin(2 * np.pi * 1 * time),  # 1 Hz oscillation
            'AccY': np.sin(2 * np.pi * 2 * time),  # 2 Hz oscillation
            'AccZ': np.sin(2 * np.pi * 0.5 * time)  # 0.5 Hz oscillation
        }
        return motion_data
    
    @pytest.fixture
    def eeg_data(self):
        """Create sample EEG data."""
        sample_data_folder = mne.datasets.sample.data_path()
        raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.pick_types(meg=False, eeg=True)
        return raw
        
    def test_motion_preprocessing(self, config, sample_motion_data, eeg_data):
        """Test motion data preprocessing."""
        data = PyMoBIData(eeg_data, motion_data=sample_motion_data)
        processor = MotionProcessor(config)
        
        processed_data = processor.run(data)
        
        # Test lowpass filtering
        for channel in processed_data.motion_data.keys():
            if not channel.endswith(('_vel', '_acc')):  # Original channels
                # Check if high frequencies are removed
                fft = np.fft.fft(processed_data.motion_data[channel])
                freqs = np.fft.fftfreq(len(fft), 1/processed_data.mne_raw.info['sfreq'])
                high_freq_power = np.mean(np.abs(fft[np.abs(freqs) > config.lowpass_motion]))
                assert high_freq_power < 0.1
                
    def test_gait_event_detection(self, config, sample_motion_data, eeg_data):
        """Test gait event detection."""
        data = PyMoBIData(eeg_data, motion_data=sample_motion_data)
        processor = MotionProcessor(config)
        
        processed_data = processor.run(data)
        
        # Test event detection
        assert processed_data.gait_events is not None
        assert len(processed_data.gait_events) > 0
        
        # Test event structure
        for event in processed_data.gait_events:
            assert 'timestamp' in event
            assert 'type' in event
            
    def test_motion_eeg_sync(self, config, sample_motion_data, eeg_data):
        """Test synchronization between motion and EEG data."""
        data = PyMoBIData(eeg_data, motion_data=sample_motion_data)
        processor = MotionProcessor(config)
        
        processed_data = processor.run(data)
        
        # Check if events are added to EEG data
        assert len(processed_data.mne_raw.annotations) > 0
        
        # Check event timing alignment
        eeg_events = processed_data.mne_raw.annotations
        motion_events = processed_data.gait_events
        
        assert len(eeg_events) == len(motion_events)
        
    def test_derivative_computation(self, config, sample_motion_data, eeg_data):
        """Test motion derivative computation."""
        data = PyMoBIData(eeg_data, motion_data=sample_motion_data)
        processor = MotionProcessor(config)
        
        processed_data = processor.run(data)
        
        # Check velocity computation
        for channel in sample_motion_data.keys():
            assert f"{channel}_vel" in processed_data.motion_data
            
        # Check acceleration computation
        for channel in sample_motion_data.keys():
            assert f"{channel}_acc" in processed_data.motion_data
            
    @pytest.mark.parametrize("sample_rate", [100, 200, 500])
    def test_different_sampling_rates(self, config, eeg_data, sample_rate):
        """Test handling of different sampling rates."""
        # Create motion data with different sampling rates
        time = np.arange(0, 10, 1/sample_rate)
        motion_data = {
            'AccX': np.sin(2 * np.pi * 1 * time)
        }
        
        data = PyMoBIData(eeg_data, motion_data=motion_data)
        processor = MotionProcessor(config)
        
        processed_data = processor.run(data)
        
        # Check if data is properly resampled
        expected_length = int(10 * eeg_data.info['sfreq'])
        actual_length = len(next(iter(processed_data.motion_data.values())))
        assert abs(expected_length - actual_length) <= 1