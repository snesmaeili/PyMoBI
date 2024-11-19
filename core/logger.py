# pymobi/core/logger.py

import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib
from core.config import PyMoBIConfig

class PyMoBILogger:
    """Advanced logging system for PyMoBI."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize logger with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing logging parameters
        """
        self.config = config
        self.processing_history = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path(self.config.study_folder) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging format
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"pymobi_{datetime.now():%Y%m%d_%H%M%S}.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('PyMoBI')
        
    def log_processing_step(self, 
                          step_name: str, 
                          params: Dict[str, Any],
                          data_hash: Optional[str] = None) -> None:
        """
        Log a processing step with parameters and data hash.
        
        Parameters
        ----------
        step_name : str
            Name of the processing step
        params : Dict[str, Any]
            Parameters used in the processing step
        data_hash : Optional[str]
            Hash of the data state after processing
        """
        step_info = {
            'step': step_name,
            'params': params,
            'timestamp': datetime.now().isoformat(),
            'data_hash': data_hash
        }
        
        self.processing_history.append(step_info)
        self.logger.info(f"Processing step: {step_name}")
        self.logger.debug(f"Parameters: {params}")
        
        # Save processing history
        self._save_processing_history()
        
    def compute_data_hash(self, data: np.ndarray) -> str:
        """
        Compute hash of data array for tracking changes.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to hash
            
        Returns
        -------
        str
            Hash string of the data
        """
        return hashlib.md5(data.tobytes()).hexdigest()
        
    def check_processing_status(self, 
                              subject_id: int,
                              required_steps: List[str]) -> bool:
        """
        Check if required processing steps have been completed.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier
        required_steps : List[str]
            List of required processing step names
            
        Returns
        -------
        bool
            True if all required steps are completed
        """
        history = self._load_processing_history(subject_id)
        if not history:
            return False
            
        completed_steps = {step['step'] for step in history}
        return all(step in completed_steps for step in required_steps)
        
    def _save_processing_history(self):
        """Save processing history to JSON file."""
        history_file = (Path(self.config.study_folder) / 
                       'logs' / 'processing_history.json')
        
        with open(history_file, 'w') as f:
            json.dump(self.processing_history, f, indent=2)
            
    def _load_processing_history(self, subject_id: int) -> List[Dict]:
        """
        Load processing history for a subject.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier
            
        Returns
        -------
        List[Dict]
            List of processing steps for the subject
        """
        history_file = (Path(self.config.study_folder) / 
                       'logs' / f'sub-{subject_id}_history.json')
        
        if not history_file.exists():
            return []
            
        with open(history_file, 'r') as f:
            return json.load(f)
            
    def get_processing_summary(self, subject_id: Optional[int] = None) -> Dict:
        """
        Get summary of processing steps.
        
        Parameters
        ----------
        subject_id : Optional[int]
            Subject identifier for specific subject summary
            
        Returns
        -------
        Dict
            Summary of processing steps
        """
        if subject_id is not None:
            history = self._load_processing_history(subject_id)
        else:
            history = self.processing_history
            
        return {
            'total_steps': len(history),
            'steps': [step['step'] for step in history],
            'timestamps': [step['timestamp'] for step in history]
        }