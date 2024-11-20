# tests/test_preprocessing.py

import pytest
import mne
import numpy as np
from pymobi import PyMoBIConfig, PyMoBIData
from pymobi.preprocessing import BasicPreprocessing, ArtifactRemoval

@pytest.fixture
def sample_data():
    """Fixture to provide sample EEG data."""
    # Load sample data from MNE
    sample_data_folder = mne.datasets.sample.data_path()
    raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    # Create PyMoBI data container
    data = PyMoBIData(raw, subject_id=1)
    return data

@pytest.fixture
def config():
    """Fixture to provide sample configuration."""
    return PyMoBIConfig(
        study_folder='test_data',
        resample_freq=250.0,
        channels_to_remove=[],
        ref_channel='FCz'
    )

def test_basic_preprocessing(sample_data, config):
    """Test basic preprocessing steps."""
    basic_proc = BasicPreprocessing(config)
    processed_data = basic_proc.run(sample_data)
    
    # Test resampling
    assert processed_data.mne_raw.info['sfreq'] == config.resample_freq
    
    # Test reference channel
    assert config.ref_channel in processed_data.mne_raw.ch_names

def test_artifact_removal(sample_data, config):
    """Test artifact removal functionality."""
    artifact_removal = ArtifactRemoval(config)
    processed_data = artifact_removal.run(sample_data)
    
    # Test bad channel detection
    assert hasattr(processed_data, 'bad_channels')
    assert isinstance(processed_data.bad_channels, list)

def test_pipeline_integration(sample_data, config):
    """Test full pipeline integration."""
    from pymobi.preprocessing import create_mobile_pipeline
    
    pipeline = create_mobile_pipeline(config)
    processed_data = pipeline.run(sample_data)
    
    # Test processing history
    assert len(processed_data.processing_history) > 0
    
    # Test data quality metrics
    assert hasattr(processed_data, 'quality_metrics')