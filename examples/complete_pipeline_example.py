# examples/complete_pipeline_example.py

import mne
import numpy as np
from pathlib import Path


def run_complete_pipeline_example():
    """
    Complete example showing the full PyMoBI pipeline functionality.
    Uses sample data and demonstrates all major features.
    """
    # Create configuration
    config = PyMoBIConfig(
        study_folder=Path("example_study"),
        filename_prefix="sub-",
        resample_freq=250.0,
        channels_to_remove=[],
        eog_channels=['EOG_l', 'EOG_r'],
        ref_channel='FCz',
        
        # Channel detection parameters
        chancorr_crit=0.8,
        chan_max_broken_time=0.3,
        chan_detect_num_iter=20,
        
        # AMICA parameters
        filter_lowCutoffFreqAMICA=1.75,
        num_models=1,
        max_threads=8,
        amica_autoreject=True,
        amica_n_rej=10,
        
        # ICLabel settings
        iclabel_classifier='lite',
        iclabel_classes=[1],
        iclabel_threshold=-1,
        
        # Final filtering
        final_filter_lower_edge=0.2,
        
        # Processing control
        save_intermediate=True
    )
    
    # Load sample data
    raw = load_sample_data()
    
    # Create data container
    data = PyMoBIData(raw, subject_id=1)
    
    # Create and run pipeline
    pipeline = create_default_pipeline(config)
    processed_data = pipeline.run(data)
    
    # Generate visualizations
    visualizer = SignalVisualizer(config)
    visualizer.plot_data_overview(processed_data)
    
    # Generate processing report
    report = ProcessingReport(config)
    report.generate_report(processed_data)
    
    return processed_data

def load_sample_data():
    """Load sample EEG data."""
    sample_data_folder = mne.datasets.sample.data_path()
    raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    # Keep only EEG channels
    raw.pick_types(meg=False, eeg=True, eog=True)
    
    return raw

if __name__ == '__main__':
    # Run complete pipeline example
    processed_data = run_complete_pipeline_example()