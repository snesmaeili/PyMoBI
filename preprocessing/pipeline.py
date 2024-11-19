# pymobi/preprocessing/pipeline.py

import mne
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData
from .basic import BasicPreprocessing
from .artifacts import ArtifactRemoval
from .ica import ICAProcessor

class PreprocessingPipeline:
    """Main preprocessing pipeline for PyMoBI."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize preprocessing pipeline.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing all processing parameters
        """
        self.config = config
        self.validate_config()
        
    def validate_config(self):
        """Validate configuration settings and create folders."""
        self.config.validate()
        self.config.create_folders()
        
    def process_subjects(self, subjects: List[int], force_recompute: bool = False):
        """
        Process multiple subjects through the pipeline.
        
        Parameters
        ----------
        subjects : List[int]
            List of subject numbers to process
        force_recompute : bool
            Whether to force recomputation of existing results
        """
        for subject in subjects:
            print(f"Processing subject {subject}")
            try:
                self._process_single_subject(subject, force_recompute)
            except Exception as e:
                print(f"Error processing subject {subject}: {str(e)}")
                
    def _process_single_subject(self, subject: int, force_recompute: bool):
        """Process a single subject through the complete pipeline."""
        
        # Setup paths
        input_path = self._get_input_path(subject)
        output_path = self._get_output_path(subject)
        
        # Check if already processed
        if not force_recompute and self._check_processed(output_path):
            print(f"Subject {subject} already processed.")
            return
            
        # Load raw data
        raw_data = self._load_raw_data(input_path)
        data = PyMoBIData(raw_data, subject_id=subject)
        
        # Remove non-experiment segments
        data = self._remove_non_exp_segments(data)
        
        # Run basic preprocessing
        data = self._run_basic_preprocessing(data)
        self._save_intermediate(data, "basic_prepared", output_path)
        
        # Handle bad channels
        data = self._handle_bad_channels(data)
        self._save_intermediate(data, "preprocessed", output_path)
        
        # Run AMICA
        data = self._run_amica_processing(data)
        self._save_intermediate(data, "amica", output_path)
        
        # Run final cleaning
        data = self._run_final_cleaning(data)
        
        # Save final results
        self._save_results(data, output_path)
        
    def _run_basic_preprocessing(self, data: PyMoBIData) -> PyMoBIData:
        """Run basic preprocessing steps."""
        basic_proc = BasicPreprocessing(self.config)
        return basic_proc.run(data)
        
    def _handle_bad_channels(self, data: PyMoBIData) -> PyMoBIData:
        """Handle bad channel detection and interpolation."""
        artifact_removal = ArtifactRemoval(self.config)
        data = artifact_removal.detect_bad_channels(data)
        return artifact_removal.interpolate_channels(data)
        
    def _run_amica_processing(self, data: PyMoBIData) -> PyMoBIData:
        """Run AMICA processing."""
        ica_proc = ICAProcessor(self.config)
        
        # High-pass filter for AMICA
        data.mne_raw.filter(
            l_freq=self.config.filter_lowCutoffFreqAMICA,
            h_freq=self.config.filter_highCutoffFreqAMICA
        )
        
        # Run AMICA
        data = ica_proc.run_amica(data)
        
        # Run ICLabel if configured
        if self.config.iclabel_classifier:
            data = ica_proc.run_iclabel(data)
            
        return data
        
    def _run_final_cleaning(self, data: PyMoBIData) -> PyMoBIData:
        """Run final cleaning steps."""
        # Apply final filtering
        data.mne_raw.filter(
            l_freq=self.config.final_filter_lower_edge,
            h_freq=self.config.final_filter_higher_edge
        )
        
        # Remove artifact ICs
        artifact_removal = ArtifactRemoval(self.config)
        return artifact_removal.remove_artifact_ics(data)
        
    def _remove_non_exp_segments(self, data: PyMoBIData) -> PyMoBIData:
        """Remove non-experiment segments from data."""
        events = self._find_exp_events(data)
        segments = self._create_segments(events)
        
        # Add buffer for time-frequency analysis
        sfreq = data.mne_raw.info['sfreq']
        buffer_samples = int(sfreq)  # 1-second buffer
        
        # Create annotation for segments to remove
        bad_segments = []
        for i, (start, end) in enumerate(segments):
            if i == 0:
                bad_segments.append([0, start - buffer_samples])
            else:
                bad_segments.append([segments[i-1][1] + buffer_samples, start - buffer_samples])
                
        # Add final segment if needed
        if segments[-1][1] < data.mne_raw.n_times:
            bad_segments.append([segments[-1][1] + buffer_samples, data.mne_raw.n_times])
            
        # Remove segments
        data.mne_raw.annotations.append(
            onset=bad_segments[:, 0] / sfreq,
            duration=(bad_segments[:, 1] - bad_segments[:, 0]) / sfreq,
            description=['bad_segment'] * len(bad_segments)
        )
        
        return data
        
    def _find_exp_events(self, data: PyMoBIData) -> List[Dict]:
        """Find experiment start/end events."""
        events = []
        for annotation in data.mne_raw.annotations:
            if 'START' in annotation['description'] and 'TEST' not in annotation['description']:
                events.append({
                    'type': 'start',
                    'latency': annotation['onset']
                })
            elif 'END' in annotation['description'] and 'test' not in annotation['description']:
                events.append({
                    'type': 'end',
                    'latency': annotation['onset']
                })
        return events
        
    def _create_segments(self, events: List[Dict]) -> np.ndarray:
        """Create segments from events."""
        segments = []
        for start, end in zip(events[::2], events[1::2]):
            segments.append([start['latency'], end['latency']])
        return np.array(segments)
        
    def _save_intermediate(self, data: PyMoBIData, stage: str, output_path: Path):
        """Save intermediate results."""
        if self.config.save_intermediate:
            filename = output_path / f"{stage}.fif"
            data.save(filename)
            
    def _save_results(self, data: PyMoBIData, output_path: Path):
        """Save final results and generate report."""
        # Save processed data
        filename = output_path / "preprocessed_and_ICA_filtered.fif"
        data.save(filename)
        
        # Generate and save quality check plots
        self._save_quality_check_plots(data, output_path)
        
    @staticmethod
    def _get_input_path(subject: int) -> Path:
        """Get input file path for subject."""
        return Path(f"sub-{subject}")
        
    @staticmethod
    def _get_output_path(subject: int) -> Path:
        """Get output file path for subject."""
        return Path(f"sub-{subject}/derivatives")
        
    @staticmethod
    def _check_processed(output_path: Path) -> bool:
        """Check if subject has already been processed."""
        return (output_path / "preprocessed_and_ICA_filtered.fif").exists()
        
    @staticmethod
    def _load_raw_data(input_path: Path) -> mne.io.Raw:
        """Load raw EEG data."""
        return mne.io.read_raw_fif(input_path / "eeg" / "raw.fif", preload=True)