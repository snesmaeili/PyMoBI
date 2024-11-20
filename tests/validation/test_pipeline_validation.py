# tests/validation/test_pipeline_validation.py

import pytest
import mne
import numpy as np
from pathlib import Path
from pymobi import PyMoBIConfig, PyMoBIData, create_default_pipeline

# Define validation datasets from BeMoBIL
VALIDATION_DATASETS = [
    {
        'name': 'visualDiscrimination',
        'url': 'https://osf.io/z8h6k/',
        'subjects': ['02', '03', '04'],
        'task': 'ssvep',
        'expected_events': ['12hz', '15hz'],
        'description': 'Visual discrimination task with SSVEP'
    },
    {
        'name': 'standingWalkingAdaptation',
        'url': 'https://gin.g-node.org/v-czk/bemobil-standing-walking-adaptation',
        'subjects': ['01', '02'],
        'task': 'walking',
        'expected_events': ['start_walking', 'stop_walking'],
        'description': 'Mobile EEG during walking task'
    }
]

class TestPipelineValidation:
    """Validation tests using BeMoBIL datasets."""
    
    @pytest.fixture(scope="session")
    def validation_config(self):
        """Create validation configuration based on BeMoBIL config."""
        return PyMoBIConfig(
            study_folder=Path("validation_data"),
            filename_prefix="sub-",
            resample_freq=250.0,
            
            # Channel detection parameters from BeMoBIL
            chancorr_crit=0.8,
            chan_max_broken_time=0.3,
            chan_detect_num_iter=20,
            chan_detected_fraction_threshold=0.5,
            
            # AMICA parameters from BeMoBIL
            filter_lowCutoffFreqAMICA=1.75,
            filter_AMICA_highPassOrder=1650,
            num_models=1,
            max_threads=8,
            amica_autoreject=True,
            amica_n_rej=10,
            amica_reject_sigma_threshold=3.0,
            amica_max_iter=2000,
            
            # ICLabel settings from BeMoBIL
            iclabel_classifier='lite',
            iclabel_classes=[1],
            iclabel_threshold=-1,
            
            # Final filtering from BeMoBIL
            final_filter_lower_edge=0.2
        )
        
    @pytest.fixture(scope="session")
    def download_validation_data(self):
        """Download validation datasets."""
        validation_data = {}
        for dataset in VALIDATION_DATASETS:
            # Create dataset directory
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