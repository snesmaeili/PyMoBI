# pymobi/preprocessing/pyamica.py

import os
import subprocess
import numpy as np
import mne
from pathlib import Path
import json
import tempfile
from typing import Optional, Dict, Any, List
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class PyAMICA:
    """Python implementation of AMICA (Adaptive Mixture ICA) using compiled binary."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize PyAMICA with configuration parameters.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing AMICA parameters:
            - amica_binary_path: Path to AMICA executable
            - num_models: Number of models (default=1)
            - max_threads: Maximum number of threads
            - max_iter: Maximum iterations
            - reject_sigma_threshold: Threshold for rejection
            - n_rej: Number of rejection iterations
        """
        self.config = config
        self.binary_path = getattr(config, 'amica_binary_path', None)
        if self.binary_path is None:
            raise ValueError("AMICA binary path must be specified in config")
            
        # AMICA parameters
        self.num_models = getattr(config, 'num_models', 1)
        self.max_threads = getattr(config, 'max_threads', 8)
        self.max_iter = getattr(config, 'amica_max_iter', 2000)
        self.reject_sigma = getattr(config, 'amica_reject_sigma_threshold', 3.0)
        self.n_rej = getattr(config, 'amica_n_rej', 10)
        
    def run(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run AMICA on the EEG data.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with MNE Raw object
            
        Returns
        -------
        PyMoBIData
            Data container with AMICA results
        """
        try:
            # Create temporary directory for AMICA files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare data and parameters
                data_file = self._prepare_data(data, temp_dir)
                param_file = self._create_param_file(data, temp_dir)
                
                # Run AMICA
                self._run_amica_binary(temp_dir, param_file)
                
                # Load results
                weights, sphere, bias = self._load_amica_results(temp_dir)
                
                # Apply results to data
                data = self._apply_amica_results(data, weights, sphere, bias)
                
                # Record processing step
                data.add_processing_step('amica', {
                    'num_models': self.num_models,
                    'max_iter': self.max_iter,
                    'reject_sigma': self.reject_sigma,
                    'n_rej': self.n_rej
                })
                
        except Exception as e:
            print(f"Error during AMICA processing: {str(e)}")
            raise
            
        return data
        
    def _prepare_data(self, data: PyMoBIData, temp_dir: str) -> str:
        """Prepare data for AMICA processing."""
        # Get EEG data
        eeg_data = data.mne_raw.get_data()
        
        # Save data in binary format
        data_file = os.path.join(temp_dir, 'data.fdt')
        eeg_data.astype('float64').tofile(data_file)
        
        return data_file
        
    def _create_param_file(self, data: PyMoBIData, temp_dir: str) -> str:
        """Create AMICA parameter file."""
        params = {
            'datafile': 'data.fdt',
            'outdir': temp_dir,
            'numchans': data.mne_raw.info['nchan'],
            'numframes': len(data.mne_raw.times),
            'num_models': self.num_models,
            'max_threads': self.max_threads,
            'max_iter': self.max_iter,
            'reject_sigma': self.reject_sigma,
            'do_reject': 1,
            'numrej': self.n_rej,
            'rejsig': self.reject_sigma,
            'rejint': 1,
            'numpass': 1
        }
        
        param_file = os.path.join(temp_dir, 'amicadefs.param')
        with open(param_file, 'w') as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")
                
        return param_file
        
    def _run_amica_binary(self, temp_dir: str, param_file: str):
        """Run AMICA binary."""
        cmd = [self.binary_path, param_file]
        
        try:
            subprocess.run(cmd, 
                         cwd=temp_dir,
                         check=True, 
                         capture_output=True, 
                         text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AMICA binary failed: {e.stderr}")
            
    def _load_amica_results(self, temp_dir: str) -> tuple:
        """Load AMICA results from files."""
        weights = np.fromfile(os.path.join(temp_dir, 'weights'), dtype='float64')
        sphere = np.fromfile(os.path.join(temp_dir, 'sphere'), dtype='float64')
        bias = np.fromfile(os.path.join(temp_dir, 'bias'), dtype='float64')
        
        # Reshape arrays
        n_chans = self.data_shape[0]
        weights = weights.reshape(n_chans, n_chans)
        sphere = sphere.reshape(n_chans, n_chans)
        bias = bias.reshape(n_chans, 1)
        
        return weights, sphere, bias
        
    def _apply_amica_results(self, 
                            data: PyMoBIData, 
                            weights: np.ndarray, 
                            sphere: np.ndarray, 
                            bias: np.ndarray) -> PyMoBIData:
        """Apply AMICA results to the data."""
        # Store ICA results
        data.ica_weights = weights
        data.ica_sphere = sphere
        
        # Apply ICA transformation
        unmixing_matrix = np.dot(weights, sphere)
        data.ica_components = np.dot(unmixing_matrix, data.mne_raw.get_data()) + bias
        
        return data
        
    def _plot_components(self, data: PyMoBIData):
        """Generate component plots."""
        # Implementation of component visualization
        pass