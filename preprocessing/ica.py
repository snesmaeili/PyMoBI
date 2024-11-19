# pymobi/preprocessing/ica.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class ICAProcessor:
    """ICA processing with multiple methods including AMICA."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize ICA processor with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing ICA parameters
        """
        self.config = config
        
    def run(self, data: PyMoBIData) -> PyMoBIData:
        """
        Run ICA decomposition using specified method.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with MNE Raw object
            
        Returns
        -------
        PyMoBIData
            Data container with ICA results
        """
        # Store original data for comparison
        original_data = data.mne_raw.get_data().copy()
        
        try:
            if self.config.ica_method == 'amica':
                data = self.run_amica(data)
            else:
                data = self.run_mne_ica(data)
                
            # Run ICLabel if configured
            if self.config.use_iclabel:
                data = self.run_iclabel(data)
                
            # Generate quality check plots
            self._plot_ica_components(data)
            
        except Exception as e:
            print(f"Error during ICA processing: {str(e)}")
            raise
            
        return data
        
    def run_mne_ica(self, data: PyMoBIData) -> PyMoBIData:
        """Run ICA using MNE's implementation."""
        # Initialize ICA
        ica = mne.preprocessing.ICA(
            n_components=self.config.n_components,
            random_state=self.config.random_state,
            method=self.config.ica_method
        )
        
        # Fit ICA
        ica.fit(data.mne_raw)
        
        # Store ICA solution
        data.ica = ica
        
        # Add processing step
        data.add_processing_step('ica', {
            'method': self.config.ica_method,
            'n_components': self.config.n_components
        })
        
        return data
        
    def run_amica(self, data: PyMoBIData) -> PyMoBIData:
        """Run AMICA using compiled binary."""
        try:
            # Create temporary directory for AMICA files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare data and parameters
                data_file = self._prepare_amica_data(data, temp_dir)
                param_file = self._create_amica_params(data, temp_dir)
                
                # Run AMICA binary
                self._run_amica_binary(temp_dir, param_file)
                
                # Load results
                weights, sphere, bias = self._load_amica_results(temp_dir)
                
                # Create MNE ICA object from AMICA results
                ica = self._create_mne_ica_from_amica(weights, sphere, bias)
                
                # Store ICA solution
                data.ica = ica
                
                # Add processing step
                data.add_processing_step('amica', {
                    'num_models': self.config.num_models,
                    'max_iter': self.config.amica_max_iter
                })
                
        except Exception as e:
            print(f"Error during AMICA processing: {str(e)}")
            raise
            
        return data
        
    def run_iclabel(self, data: PyMoBIData) -> PyMoBIData:
        """Run ICLabel classification."""
        if data.ica is None:
            raise ValueError("No ICA solution found. Run ICA first.")
            
        # Implementation of ICLabel classification
        # This would interface with ICLabel implementation
        pass
        
    def _prepare_amica_data(self, data: PyMoBIData, temp_dir: str) -> str:
        """Prepare data for AMICA processing."""
        # Get EEG data
        eeg_data = data.mne_raw.get_data()
        
        # Save data in binary format
        data_file = Path(temp_dir) / 'data.fdt'
        eeg_data.astype('float64').tofile(data_file)
        
        return str(data_file)
        
    def _create_amica_params(self, data: PyMoBIData, temp_dir: str) -> str:
        """Create AMICA parameter file."""
        params = {
            'datafile': 'data.fdt',
            'outdir': temp_dir,
            'numchans': data.mne_raw.info['nchan'],
            'numframes': len(data.mne_raw.times),
            'num_models': self.config.num_models,
            'max_threads': self.config.max_threads,
            'max_iter': self.config.amica_max_iter,
            'reject_sigma': self.config.amica_reject_sigma_threshold,
            'do_reject': 1,
            'numrej': self.config.amica_n_rej
        }
        
        param_file = Path(temp_dir) / 'amicadefs.param'
        with open(param_file, 'w') as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")
                
        return str(param_file)
        
    def _run_amica_binary(self, temp_dir: str, param_file: str):
        """Run AMICA binary."""
        cmd = [self.config.amica_binary_path, param_file]
        
        try:
            subprocess.run(cmd, 
                         cwd=temp_dir,
                         check=True, 
                         capture_output=True, 
                         text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AMICA binary failed: {e.stderr}")
            
    def _load_amica_results(self, temp_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load AMICA results from files."""
        weights = np.fromfile(Path(temp_dir) / 'weights', dtype='float64')
        sphere = np.fromfile(Path(temp_dir) / 'sphere', dtype='float64')
        bias = np.fromfile(Path(temp_dir) / 'bias', dtype='float64')
        
        # Reshape arrays
        n_chans = weights.size // weights.size
        weights = weights.reshape(n_chans, n_chans)
        sphere = sphere.reshape(n_chans, n_chans)
        bias = bias.reshape(n_chans, 1)
        
        return weights, sphere, bias
        
    def _create_mne_ica_from_amica(self, 
                                  weights: np.ndarray, 
                                  sphere: np.ndarray, 
                                  bias: np.ndarray) -> mne.preprocessing.ICA:
        """Create MNE ICA object from AMICA results."""
        ica = mne.preprocessing.ICA(n_components=weights.shape[0])
        ica.unmixing_matrix_ = np.dot(weights, sphere)
        ica.n_components_ = weights.shape[0]
        
        return ica
        
    def _plot_ica_components(self, data: PyMoBIData):
        """Generate component plots."""
        if data.ica is None:
            return
            
        # Create figure for component properties
        fig = data.ica.plot_components(
            picks=range(20),  # Plot first 20 components
            show=False
        )
        
        # Save figure
        if hasattr(data, 'subject_id'):
            output_path = self._get_output_path(data)
            plt.savefig(output_path / f'sub-{data.subject_id}_ica_components.png')
        plt.close()
        
    def _get_output_path(self, data: PyMoBIData) -> Path:
        """Get output path for plots."""
        base_path = Path(self.config.study_folder) / self.config.spatial_filters_folder
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"