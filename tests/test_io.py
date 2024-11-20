# tests/test_io.py

import pytest
import mne
import numpy as np
from pathlib import Path
from pymobi import PyMoBIConfig, PyMoBIData
from pymobi.io import DataReader, BIDSWriter

class TestDataIO:
    """Test suite for data input/output functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PyMoBIConfig(
            study_folder=Path("test_data"),
            filename_prefix="sub-",
            resample_freq=250.0,
            channels_to_remove=[],
            ref_channel='FCz'
        )
    
    @pytest.fixture
    def sample_raw(self):
        """Create sample raw data."""
        # Create synthetic data
        data = np.random.randn(32, 10000)  # 32 channels, 10000 timepoints
        sfreq = 1000.0
        ch_names = [f'EEG{i:03d}' for i in range(32)]
        ch_types = ['eeg'] * 32
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        return mne.io.RawArray(data, info)
    
    def test_bids_conversion(self, config, sample_raw, tmp_path):
        """Test BIDS format conversion."""
        # Create test data
        data = PyMoBIData(sample_raw, subject_id=1)
        
        # Set up BIDS writer
        writer = BIDSWriter(config)
        
        # Write to BIDS format
        bids_path = tmp_path / 'bids'
        writer.write_bids(data, bids_path)
        
        # Check BIDS structure
        assert (bids_path / 'sub-001').exists()
        assert (bids_path / 'sub-001' / 'eeg').exists()
        assert (bids_path / 'dataset_description.json').exists()
        
        # Read back and verify
        reader = DataReader(config)
        loaded_data = reader.read_bids(bids_path / 'sub-001')
        
        # Verify data integrity
        np.testing.assert_array_almost_equal(
            data.mne_raw.get_data(),
            loaded_data.mne_raw.get_data()
        )
        
    def test_xdf_loading(self, config, tmp_path):
        """Test XDF file loading."""
        # Create mock XDF file
        xdf_path = tmp_path / 'test.xdf'
        self._create_mock_xdf(xdf_path)
        
        # Read XDF
        reader = DataReader(config)
        data = reader.read_raw(xdf_path)
        
        # Verify data structure
        assert isinstance(data, PyMoBIData)
        assert data.mne_raw is not None
        
    def test_multiple_streams(self, config, tmp_path):
        """Test loading data with multiple streams."""
        # Create test data with EEG and motion streams
        eeg_data = np.random.randn(32, 10000)
        motion_data = np.random.randn(6, 10000)  # 6 motion channels
        
        # Save as mock data
        test_file = tmp_path / 'multistream.xdf'
        self._create_mock_multistream(test_file, eeg_data, motion_data)
        
        # Load data
        reader = DataReader(config)
        data = reader.read_raw(test_file)
        
        # Verify streams
        assert data.motion_data is not None
        assert len(data.motion_data) == 6
        
    def test_bids_metadata(self, config, sample_raw, tmp_path):
        """Test BIDS metadata handling."""
        # Create test data with metadata
        data = PyMoBIData(sample_raw, subject_id=1)
        data.metadata = {
            'task': 'walking',
            'condition': 'outdoor',
            'age': 25
        }
        
        # Write to BIDS
        writer = BIDSWriter(config)
        bids_path = tmp_path / 'bids'
        writer.write_bids(data, bids_path)
        
        # Check metadata files
        assert (bids_path / 'sub-001' / 'sub-001_task-walking_eeg.json').exists()
        
        # Read back and verify
        reader = DataReader(config)
        loaded_data = reader.read_bids(bids_path / 'sub-001')
        assert loaded_data.metadata['task'] == 'walking'
        
    def test_events_handling(self, config, sample_raw, tmp_path):
        """Test event handling in data IO."""
        # Create test events
        events = [
            {'type': 'stimulus', 'latency': 1000, 'duration': 0.1},
            {'type': 'response', 'latency': 1500, 'duration': 0},
            {'type': 'motion', 'latency': 2000, 'duration': 0.5}
        ]
        
        # Add events to data
        data = PyMoBIData(sample_raw, subject_id=1)
        for event in events:
            data.mne_raw.annotations.append(
                onset=event['latency'] / data.mne_raw.info['sfreq'],
                duration=event['duration'],
                description=event['type']
            )
        
        # Write to BIDS
        writer = BIDSWriter(config)
        bids_path = tmp_path / 'bids'
        writer.write_bids(data, bids_path)
        
        # Read back and verify events
        reader = DataReader(config)
        loaded_data = reader.read_bids(bids_path / 'sub-001')
        
        loaded_events = loaded_data.mne_raw.annotations
        assert len(loaded_events) == len(events)
        
    @staticmethod
    def _create_mock_xdf(path: Path):
        """Create a mock XDF file for testing."""
        # Implementation depends on XDF file format
        pass
        
    @staticmethod
    def _create_mock_multistream(path: Path, eeg_data: np.ndarray, motion_data: np.ndarray):
        """Create a mock multi-stream data file."""
        # Implementation depends on file format
        pass