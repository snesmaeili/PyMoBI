# tests/validation/test_open_datasets.py

import pytest
import mne
import numpy as np
from pathlib import Path
from pymobi import PyMoBIConfig, PyMoBIData, create_default_pipeline

# Define open datasets for validation
VALIDATION_DATASETS = [
    {
        'name': 'visual_discrimination',
        'url': 'https://osf.io/download/z8h6k/',
        'subjects': ['02', '03', '04'],
        'task': 'ssvep',
        'expected_events': ['12hz', '15hz']
    },
    {
        'name': 'mobile_brain_body',
        'url': 'https://gin.g-node.org/v-czk/bemobil-standing-walking-adaptation',
        'subjects': ['01', '02'],
        'task': 'walking',
        'expected_events': ['start_walking', 'stop_walking']
    }
]

class TestValidation:
    """Validation tests using open datasets."""
    
    @pytest.fixture(scope="session")
    def validation_config(self):
        """Create validation configuration."""
        return PyMoBIConfig(
            study_folder=Path("validation_data"),
            filename_prefix="sub-",
            resample_freq=250.0,
            channels_to_remove=[],
            ref_channel='FCz',
            
            # ASR parameters
            asr_cutoff=20,
            use_asr=True,
            
            # AMICA parameters
            filter_lowCutoffFreqAMICA=1.75,
            num_models=1,
            max_threads=4,
            
            # Save intermediate results
            save_intermediate=True
        )
        
    @pytest.fixture(scope="session")
    def download_validation_data(self):
        """Download validation datasets."""
        validation_data = {}
        for dataset in VALIDATION_DATASETS:
            # Download dataset
            outdir = Path('validation_data') / dataset['name']
            outdir.mkdir(parents=True, exist_ok=True)
            
            dataset_data = {}
            for subject in dataset['subjects']:
                # Download subject data
                fname = outdir / f"sub-{subject}_task-{dataset['task']}_eeg.vhdr"
                if not fname.exists():
                    mne.utils.download(
                        f"{dataset['url']}/sub-{subject}/eeg/{fname.name}",
                        fname
                    )
                
                # Load data
                raw = mne.io.read_raw_brainvision(fname, preload=True)
                dataset_data[subject] = raw
                
            validation_data[dataset['name']] = dataset_data
            
        return validation_data
        
    @pytest.mark.parametrize("dataset", VALIDATION_DATASETS)
    def test_pipeline_validation(self, dataset, validation_config, download_validation_data):
        """Validate pipeline on open datasets."""
        dataset_data = download_validation_data[dataset['name']]
        
        for subject in dataset['subjects']:
            # Create PyMoBI data container
            raw = dataset_data[subject]
            data = PyMoBIData(raw, subject_id=int(subject))
            
            # Run pipeline
            pipeline = create_default_pipeline(validation_config)
            processed_data = pipeline.run(data)
            
            # Validate results
            self._validate_preprocessing(processed_data)
            self._validate_artifact_removal(processed_data)
            self._validate_ica(processed_data)
            self._validate_events(processed_data, dataset['expected_events'])
            
    def _validate_preprocessing(self, data: PyMoBIData):
        """Validate preprocessing results."""
        # Check sampling rate
        assert data.mne_raw.info['sfreq'] == 250.0
        
        # Check channel properties
        assert all(ch.startswith('EEG') or ch in ['EOG_l', 'EOG_r', 'FCz'] 
                  for ch in data.mne_raw.ch_names)
        
        # Check data quality
        assert not np.any(np.isnan(data.mne_raw.get_data()))
        assert not np.any(np.isinf(data.mne_raw.get_data()))
        
    def _validate_artifact_removal(self, data: PyMoBIData):
        """Validate artifact removal results."""
        # Check bad channels
        assert hasattr(data, 'bad_channels')
        assert len(data.bad_channels) < len(data.mne_raw.ch_names) * 0.2
        
        # Check ASR results if used
        if hasattr(data, 'asr_removed_samples'):
            assert data.asr_removed_samples < len(data.mne_raw.times) * 0.3
            
    def _validate_ica(self, data: PyMoBIData):
        """Validate ICA results."""
        # Check ICA attributes
        assert data.ica is not None
        assert hasattr(data.ica, 'unmixing_matrix_')
        
        # Check component properties
        if hasattr(data, 'iclabel_scores'):
            brain_components = np.where(data.iclabel_scores[:, 0] > 0.5)[0]
            assert len(brain_components) > len(data.ica.unmixing_matrix_) * 0.3
            
    def _validate_events(self, data: PyMoBIData, expected_events: List[str]):
        """Validate event detection."""
        # Check if all expected events are present
        event_types = [event['type'] for event in data.mne_raw.annotations]
        for event_type in expected_events:
            assert any(event_type in evt for evt in event_types)