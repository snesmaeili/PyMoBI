# tests/test_integration.py

import pytest
import mne
import numpy as np
from pathlib import Path
from pymobi import PyMoBIConfig, PyMoBIData, create_default_pipeline

# Define test datasets
DATASETS = [
    {
        'name': 'visual_discrimination',
        'url': 'https://osf.io/download/z8h6k/',
        'subject': '02',
        'task': 'ssvep'
    },
    {
        'name': 'mobile_brain_body',
        'url': 'https://gin.g-node.org/v-czk/bemobil-standing-walking-adaptation',
        'subject': '01',
        'task': 'walking'
    }
]

@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    return PyMoBIConfig(
        study_folder=Path("test_data"),
        filename_prefix="sub-",
        resample_freq=250.0,
        channels_to_remove=[],
        eog_channels=['EOG_l', 'EOG_r'],
        ref_channel='FCz',
        
        # Channel detection parameters
        chancorr_crit=0.8,
        chan_max_broken_time=0.3,
        chan_detect_num_iter=20,
        
        # ASR parameters
        asr_cutoff=20,
        use_asr=True,
        
        # AMICA parameters
        filter_lowCutoffFreqAMICA=1.75,
        num_models=1,
        max_threads=4,
        amica_autoreject=True,
        
        # Save intermediate results
        save_intermediate=True
    )

@pytest.fixture(scope="session")
def download_test_data():
    """Download test datasets."""
    test_data = {}
    for dataset in DATASETS:
        # Download and load dataset
        raw = _download_and_load_dataset(dataset)
        test_data[dataset['name']] = raw
    return test_data

def _download_and_load_dataset(dataset_info):
    """Download and load a specific dataset."""
    outdir = Path('test_data') / dataset_info['name']
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Download if not exists
    fname = outdir / f"sub-{dataset_info['subject']}_task-{dataset_info['task']}_eeg.vhdr"
    if not fname.exists():
        mne.utils.download(dataset_info['url'], fname)
    
    # Load data
    raw = mne.io.read_raw_brainvision(fname, preload=True)
    return raw

@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_complete_pipeline(dataset, test_config, download_test_data):
    """Test complete pipeline on real datasets."""
    # Get raw data
    raw = download_test_data[dataset['name']]
    
    # Create PyMoBI data container
    data = PyMoBIData(raw, subject_id=int(dataset['subject']))
    
    # Create and run pipeline
    pipeline = create_default_pipeline(test_config)
    processed_data = pipeline.run(data)
    
    # Basic assertions
    assert processed_data.mne_raw.info['sfreq'] == test_config.resample_freq
    assert len(processed_data.processing_history) > 0
    assert hasattr(processed_data, 'bad_channels')
    assert processed_data.ica is not None

def test_preprocessing_quality(test_config, download_test_data):
    """Test quality of preprocessing results."""
    # Use visual discrimination dataset
    raw = download_test_data['visual_discrimination']
    data = PyMoBIData(raw, subject_id=2)
    
    # Run pipeline
    pipeline = create_default_pipeline(test_config)
    processed_data = pipeline.run(data)
    
    # Test data quality metrics
    _verify_data_quality(processed_data)
    
    # Test artifact removal
    _verify_artifact_removal(processed_data)
    
    # Test ICA quality
    _verify_ica_quality(processed_data)

def _verify_data_quality(data: PyMoBIData):
    """Verify quality of processed data."""
    # Check sampling rate
    assert data.mne_raw.info['sfreq'] == 250.0
    
    # Check for bad channels
    assert hasattr(data, 'bad_channels')
    assert len(data.bad_channels) < len(data.mne_raw.ch_names) * 0.2
    
    # Check data rank
    rank = np.linalg.matrix_rank(data.mne_raw.get_data())
    assert rank > len(data.mne_raw.ch_names) * 0.8

def _verify_artifact_removal(data: PyMoBIData):
    """Verify artifact removal quality."""
    # Get data array
    eeg_data = data.mne_raw.get_data()
    
    # Check for extreme values
    assert np.abs(eeg_data).max() < 150  # μV
    
    # Check for flat channels
    var_channels = np.var(eeg_data, axis=1)
    assert not np.any(var_channels < 0.1)

def _verify_ica_quality(data: PyMoBIData):
    """Verify ICA decomposition quality."""
    # Check ICA attributes
    assert data.ica is not None
    assert hasattr(data.ica, 'unmixing_matrix_')
    
    # Check component properties
    if hasattr(data, 'iclabel_scores'):
        brain_components = np.where(data.iclabel_scores[:, 0] > 0.5)[0]
        assert len(brain_components) > len(data.ica.unmixing_matrix_) * 0.3

@pytest.mark.parametrize("dataset", DATASETS)
def test_motion_integration(dataset, test_config, download_test_data):
    """Test motion data integration."""
    # Get raw data
    raw = download_test_data[dataset['name']]
    
    # Add motion data if available
    motion_data = _load_motion_data(dataset)
    
    # Create data container
    data = PyMoBIData(raw, subject_id=int(dataset['subject']), motion_data=motion_data)
    
    # Run pipeline
    pipeline = create_default_pipeline(test_config)
    processed_data = pipeline.run(data)
    
    # Verify motion integration
    if motion_data is not None:
        assert processed_data.motion_data is not None
        assert processed_data.gait_events is not None

def _load_motion_data(dataset_info):
    """Load motion data if available."""
    motion_file = Path('test_data') / dataset_info['name'] / f"sub-{dataset_info['subject']}_task-{dataset_info['task']}_motion.tsv"
    if motion_file.exists():
        return np.loadtxt(motion_file)
    return None
