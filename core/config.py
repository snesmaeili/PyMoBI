# pymobi/core/config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class PyMoBIConfig:
    """Configuration management for PyMoBI preprocessing."""
    
    # Required parameters
    study_folder: Path
    
    # File structure settings
    filename_prefix: str = "sub-"
    
    # Folder structure with default values
    source_data_folder: Path = field(default_factory=lambda: Path("0_source-data"))
    bids_data_folder: Path = field(default_factory=lambda: Path("1_BIDS-data"))
    raw_eeglab_data_folder: Path = field(default_factory=lambda: Path("2_raw-EEGLAB"))
    eeg_preprocessing_folder: Path = field(default_factory=lambda: Path("3_EEG-preprocessing"))
    spatial_filters_folder: Path = field(default_factory=lambda: Path("4_spatial-filters"))
    spatial_filters_amica_folder: Path = field(default_factory=lambda: Path("4-1_AMICA"))
    single_subject_analysis_folder: Path = field(default_factory=lambda: Path("5_single-subject-EEG-analysis"))
    motion_analysis_folder: Path = field(default_factory=lambda: Path("6_single-subject-motion-analysis"))
    
    # File naming conventions
    merged_filename: str = "merged_EEG.fif"
    basic_prepared_filename: str = "basic_prepared.fif"
    preprocessed_filename: str = "preprocessed.fif"
    filtered_filename: str = "filtered.fif"
    amica_filename_output: str = "AMICA.fif"
    dipfitted_filename: str = "dipfitted.fif"
    preprocessed_and_ica_filename: str = "preprocessed_and_ICA.fif"
    single_subject_cleaned_ica_filename: str = "cleaned_with_ICA.fif"
    merged_motion_filename: str = "merged_MOTION.fif"
    processed_motion_filename: str = "motion_processed.fif"
    
    # Preprocessing parameters
    channels_to_remove: List[str] = field(default_factory=list)
    eog_channels: List[str] = field(default_factory=list)
    ref_channel: Optional[str] = "FCz"
    rename_channels: Dict[str, str] = field(default_factory=dict)
    resample_freq: float = 250.0
    
    # Channel detection parameters
    chancorr_crit: float = 0.8
    chan_max_broken_time: float = 0.3
    chan_detect_num_iter: int = 20
    chan_detected_fraction_threshold: float = 0.5
    flatline_crit: str = "off"
    line_noise_crit: str = "off"
    num_chan_rej_max_target: float = 0.2
    
    # Channel locations
    channel_locations_filename: Optional[str] = None
    
    # Zapline settings
    zapline_config: Dict[str, Any] = field(default_factory=lambda: {
        "noisefreqs": [],
        "bandwidth": 2.0
    })
    
    # AMICA parameters
    filter_lowCutoffFreqAMICA: float = 1.75
    filter_AMICA_highPassOrder: int = 1650
    filter_highCutoffFreqAMICA: Optional[float] = None
    filter_AMICA_lowPassOrder: Optional[int] = None
    num_models: int = 1
    max_threads: int = 8
    
    # AMICA auto-rejection settings
    amica_autoreject: bool = True
    amica_n_rej: int = 10
    amica_reject_sigma_threshold: float = 3.0
    amica_max_iter: int = 2000
    
    # ICLabel settings
    iclabel_classifier: str = "lite"
    iclabel_classes: List[int] = field(default_factory=lambda: [1])
    iclabel_threshold: float = -1
    
    # Final filtering
    final_filter_lower_edge: float = 0.2
    final_filter_higher_edge: Optional[float] = None
    
    # Motion processing
    lowpass_motion: float = 8.0
    lowpass_motion_after_derivative: float = 24.0
    
    # Processing control
    force_recompute: bool = False
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.study_folder = Path(self.study_folder)
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str) and field_name.endswith('_folder'):
                setattr(self, field_name, Path(field_value))
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PyMoBIConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.study_folder.exists():
            raise ValueError(f"Study folder does not exist: {self.study_folder}")
        
        if self.resample_freq <= 0:
            raise ValueError(f"Invalid resample frequency: {self.resample_freq}")
        
        if self.chan_detected_fraction_threshold < 0 or self.chan_detected_fraction_threshold > 1:
            raise ValueError(f"Invalid channel detection threshold: {self.chan_detected_fraction_threshold}")
        
        return True
    
    def create_folders(self):
        """Create folder structure if it doesn't exist."""
        folders = [
            self.source_data_folder,
            self.bids_data_folder,
            self.raw_eeglab_data_folder,
            self.eeg_preprocessing_folder,
            self.spatial_filters_folder,
            self.spatial_filters_amica_folder,
            self.single_subject_analysis_folder,
            self.motion_analysis_folder
        ]
        
        for folder in folders:
            full_path = self.study_folder / folder
            full_path.mkdir(parents=True, exist_ok=True)
    
    def get_subject_path(self, subject_id: int, folder: Path) -> Path:
        """Get subject-specific path."""
        return self.study_folder / folder / f"{self.filename_prefix}{subject_id}"