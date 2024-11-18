# examples/basic_preprocessing.py

import mne
from pymobi.core.config import PyMoBIConfig
from pymobi.core.data import PyMoBIData
from pymobi.preprocessing.basic import create_mobile_pipeline

def main():
    # Load sample EEG data using MNE.
    raw = mne.io.read_raw_fif('sample_data.fif')
    
    # Create PyMoBIData container.
    eeg_data = PyMoBIData(raw)
    
    # Define configuration for preprocessing.
    config = PyMoBIConfig(
        mne_preprocessing={'l_freq': 1.0, 'h_freq': 40.0},
        motion_preprocessing={},
        custom_processing={},
        visualization={}
    )
    
    # Create and run the preprocessing pipeline.
    pipeline = create_mobile_pipeline(config)
    processed_data = pipeline.run(eeg_data)
    
    # Save or visualize the processed data.
    processed_data.to_mne().save('processed_eeg.fif')

if __name__ == "__main__":
    main()