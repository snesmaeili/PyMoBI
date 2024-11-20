# pymobi/gui/nodes/amica.py

from typing import Dict, Any, List
import numpy as np
import mne
import subprocess
from pathlib import Path
import tempfile
from ..core import BaseNode

class AMICANode(BaseNode):
    """Node for AMICA (Adaptive Mixture ICA) processing."""
    
    def __init__(self):
        super().__init__("AMICA")
        self.inputs = {'data': None}
        self.outputs = {'data': None}
        self.parameters = {
            'num_models': {'type': 'int', 'value': 1},
            'max_threads': {'type': 'int', 'value': 8},
            'max_iter': {'type': 'int', 'value': 2000},
            'reject_sigma': {'type': 'float', 'value': 3.0},
            'n_rej': {'type': 'int', 'value': 10},
            'filter_low': {'type': 'float', 'value': 1.75},
            'filter_high': {'type': 'float', 'value': None},
            'autoreject': {'type': 'bool', 'value': True}
        }
        
    def process(self) -> Dict[str, Any]:
        """Run AMICA processing."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        
        # High-pass filter for AMICA
        if self.parameters['filter_low']['value']:
            data.mne_raw.filter(
                l_freq=self.parameters['filter_low']['value'],
                h_freq=self.parameters['filter_high']['value']
            )
            
        try:
            # Create temporary directory for AMICA files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare data and parameters
                data_file = self._prepare_amica_data(data, temp_dir)
                param_file = self._create_amica_params(data, temp_dir)
                
                # Run AMICA
                self._run_amica_binary(temp_dir, param_file)
                
                # Load results
                weights, sphere, bias = self._load_amica_results(temp_dir)
                
                # Create MNE ICA object
                ica = self._create_mne_ica(weights, sphere, bias)
                
                # Store ICA solution
                data.ica = ica
                
                # Run autorejection if enabled
                if self.parameters['autoreject']['value']:
                    self._run_autorejection(data)
                    
        except Exception as e:
            print(f"Error during AMICA processing: {str(e)}")
            raise
            
        self.outputs['data'] = data
        return {'data': data}
        
    def _prepare_amica_data(self, data: Any, temp_dir: str) -> str:
        """Prepare data for AMICA processing."""
        # Get EEG data
        eeg_data = data.mne_raw.get_data()
        
        # Save data in binary format
        data_file = Path(temp_dir) / 'data.fdt'
        eeg_data.astype('float64').tofile(data_file)
        
        return str(data_file)
        
    def _create_amica_params(self, data: Any, temp_dir: str) -> str:
        """Create AMICA parameter file."""
        params = {
            'datafile': 'data.fdt',
            'outdir': temp_dir,
            'numchans': data.mne_raw.info['nchan'],
            'numframes': len(data.mne_raw.times),
            'num_models': self.parameters['num_models']['value'],
            'max_threads': self.parameters['max_threads']['value'],
            'max_iter': self.parameters['max_iter']['value'],
            'reject_sigma': self.parameters['reject_sigma']['value'],
            'do_reject': 1 if self.parameters['autoreject']['value'] else 0,
            'numrej': self.parameters['n_rej']['value']
        }
        
        param_file = Path(temp_dir) / 'amicadefs.param'
        with open(param_file, 'w') as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")
                
        return str(param_file)
        
    def _run_amica_binary(self, temp_dir: str, param_file: str):
        """Run AMICA binary."""
        cmd = ['amica15mkl', param_file]
        
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
        weights = np.fromfile(Path(temp_dir) / 'weights', dtype='float64')
        sphere = np.fromfile(Path(temp_dir) / 'sphere', dtype='float64')
        bias = np.fromfile(Path(temp_dir) / 'bias', dtype='float64')
        
        # Reshape arrays
        n_chans = weights.size // weights.size
        weights = weights.reshape(n_chans, n_chans)
        sphere = sphere.reshape(n_chans, n_chans)
        bias = bias.reshape(n_chans, 1)
        
        return weights, sphere, bias
        
    def _create_mne_ica(self, weights: np.ndarray, 
                       sphere: np.ndarray, 
                       bias: np.ndarray) -> mne.preprocessing.ICA:
        """Create MNE ICA object from AMICA results."""
        ica = mne.preprocessing.ICA(n_components=weights.shape[0])
        ica.unmixing_matrix_ = np.dot(weights, sphere)
        ica.n_components_ = weights.shape[0]
        
        return ica
        
    def _run_autorejection(self, data: Any):
        """Run AMICA autorejection."""
        if not hasattr(data, 'ica'):
            raise ValueError("No ICA solution found")
            
        # Get data
        eeg_data = data.mne_raw.get_data()
        
        # Apply current ICA
        ica_data = data.ica.get_sources(data.mne_raw).get_data()
        
        # Compute rejection threshold
        sigma = np.std(ica_data, axis=1)
        threshold = self.parameters['reject_sigma']['value'] * sigma[:, None]
        
        # Find bad samples
        bad_samples = np.any(np.abs(ica_data) > threshold, axis=0)
        
        # Store rejection info
        data.metadata['bad_samples'] = bad_samples
        data.metadata['bad_samples_percent'] = (np.sum(bad_samples) / 
                                              len(bad_samples) * 100)
        
    def required_inputs(self) -> List[str]:
        return ['data']