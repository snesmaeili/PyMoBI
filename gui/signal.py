# pymobi/gui/nodes/signal.py

from typing import Dict, Any, List, Optional
import numpy as np
import mne
from scipy import signal
from ..core import BaseNode

class SignalProcessingNode(BaseNode):
    """Base class for signal processing nodes."""
    
    def __init__(self, label: str):
        super().__init__(label)
        self.metadata = {}
        
    def update_metadata(self, data: np.ndarray, sfreq: float):
        """Update metadata with signal properties."""
        self.metadata.update({
            'sampling_rate': sfreq,
            'shape': data.shape,
            'dtype': str(data.dtype)
        })

class FilterNode(SignalProcessingNode):
    """Node for filtering signals."""
    
    def __init__(self):
        super().__init__("Filter")
        self.inputs = {'data': None}
        self.outputs = {'filtered_data': None}
        self.parameters = {
            'lowpass': {'type': 'float', 'value': 40.0},
            'highpass': {'type': 'float', 'value': 1.0},
            'notch': {'type': 'float', 'value': 50.0},
            'filter_method': {
                'type': 'select',
                'value': 'fir',
                'options': ['fir', 'iir']
            },
            'filter_order': {'type': 'int', 'value': 1001}
        }
        
    def process(self) -> Dict[str, Any]:
        """Apply filters to the signal."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        sfreq = data.metadata['sampling_rate']
        
        # Apply filters
        filtered_data = data.copy()
        
        # Highpass filter
        if self.parameters['highpass']['value']:
            filtered_data = self._apply_filter(
                filtered_data,
                self.parameters['highpass']['value'],
                None,
                sfreq
            )
            
        # Lowpass filter
        if self.parameters['lowpass']['value']:
            filtered_data = self._apply_filter(
                filtered_data,
                None,
                self.parameters['lowpass']['value'],
                sfreq
            )
            
        # Notch filter
        if self.parameters['notch']['value']:
            filtered_data = self._apply_notch(
                filtered_data,
                self.parameters['notch']['value'],
                sfreq
            )
            
        self.outputs['filtered_data'] = filtered_data
        return {'filtered_data': filtered_data}
        
    def _apply_filter(self, data: np.ndarray, 
                     lowcut: Optional[float], 
                     highcut: Optional[float],
                     fs: float) -> np.ndarray:
        """Apply bandpass filter."""
        nyq = 0.5 * fs
        order = self.parameters['filter_order']['value']
        
        if self.parameters['filter_method']['value'] == 'fir':
            # FIR filter
            if lowcut and highcut:
                # Bandpass
                b = signal.firwin(order, [lowcut/nyq, highcut/nyq], 
                                pass_zero=False)
            elif lowcut:
                # Highpass
                b = signal.firwin(order, lowcut/nyq, pass_zero=False)
            else:
                # Lowpass
                b = signal.firwin(order, highcut/nyq)
                
            a = [1.0]
        else:
            # IIR filter (Butterworth)
            if lowcut and highcut:
                b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], 
                                   btype='band')
            elif lowcut:
                b, a = signal.butter(order, lowcut/nyq, btype='high')
            else:
                b, a = signal.butter(order, highcut/nyq, btype='low')
                
        return signal.filtfilt(b, a, data)
        
    def _apply_notch(self, data: np.ndarray, 
                     freq: float, fs: float) -> np.ndarray:
        """Apply notch filter."""
        nyq = 0.5 * fs
        q = 30.0  # Quality factor
        w0 = freq/nyq
        b, a = signal.iirnotch(w0, q)
        return signal.filtfilt(b, a, data)
        
    def required_inputs(self) -> List[str]:
        return ['data']

class PSDNode(SignalProcessingNode):
    """Node for computing power spectral density."""
    
    def __init__(self):
        super().__init__("PSD")
        self.inputs = {'data': None}
        self.outputs = {'psd': None}
        self.parameters = {
            'method': {
                'type': 'select',
                'value': 'welch',
                'options': ['welch', 'multitaper']
            },
            'window_length': {'type': 'float', 'value': 2.0},
            'overlap': {'type': 'float', 'value': 0.5},
            'fmin': {'type': 'float', 'value': 0.0},
            'fmax': {'type': 'float', 'value': 100.0}
        }
        
    def process(self) -> Dict[str, Any]:
        """Compute power spectral density."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        sfreq = data.metadata['sampling_rate']
        
        # Compute window parameters
        win_length = int(self.parameters['window_length']['value'] * sfreq)
        n_overlap = int(win_length * self.parameters['overlap']['value'])
        
        if self.parameters['method']['value'] == 'welch':
            freqs, psd = signal.welch(
                data,
                fs=sfreq,
                nperseg=win_length,
                noverlap=n_overlap,
                nfft=None
            )
        else:
            from mne.time_frequency import psd_array_multitaper
            psd, freqs = psd_array_multitaper(
                data,
                sfreq,
                fmin=self.parameters['fmin']['value'],
                fmax=self.parameters['fmax']['value']
            )
            
        # Update metadata
        self.update_metadata(psd, sfreq)
        
        self.outputs['psd'] = psd
        return {'psd': psd}
        
    def required_inputs(self) -> List[str]:
        return ['data']