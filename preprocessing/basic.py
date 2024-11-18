# pymobi/preprocessing/basic.py

import mne

class BasicPreprocessing:
    """Basic EEG preprocessing using MNE functions."""
    
    def __init__(self, l_freq=1.0, h_freq=40.0):
        self.l_freq = l_freq  # Low frequency cut-off.
        self.h_freq = h_freq  # High frequency cut-off.
    
    def run(self, data, config):
        """Run basic preprocessing (filtering)."""
        data.mne_raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
        return data


def create_mobile_pipeline(config):
    """Create a mobile EEG preprocessing pipeline."""
    pipeline = PreprocessingPipeline()
    
    # Add basic preprocessing step.
    pipeline.add_step(BasicPreprocessing(
        l_freq=config.mne_preprocessing['l_freq'], 
        h_freq=config.mne_preprocessing['h_freq']
    ))
    
    return pipeline