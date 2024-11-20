# pymobi/gui/nodes/stream.py

from typing import Dict, Any, List, Optional
import numpy as np
import mne
from ..core import BaseNode
from queue import Queue
import threading
import time

class StreamNode(BaseNode):
    """Base class for streaming data nodes."""
    
    def __init__(self, label: str):
        super().__init__(label)
        self.stream_buffer = Queue()
        self.is_streaming = False
        self.stream_thread = None
        
    def start_streaming(self):
        """Start data streaming."""
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._stream_data)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
    def stop_streaming(self):
        """Stop data streaming."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
            
    def _stream_data(self):
        """Stream data implementation."""
        raise NotImplementedError("Subclasses must implement _stream_data")

class FileStreamNode(StreamNode):
    """Node for streaming data from file."""
    
    def __init__(self):
        super().__init__("File Stream")
        self.inputs = {}
        self.outputs = {'data': None}
        self.parameters = {
            'filename': {'type': 'str', 'value': ''},
            'chunk_size': {'type': 'float', 'value': 1.0},
            'loop': {'type': 'bool', 'value': False}
        }
        self.raw = None
        self.current_position = 0
        
    def process(self) -> Dict[str, Any]:
        """Process streaming data."""
        if not self.raw and self.parameters['filename']['value']:
            self.raw = mne.io.read_raw(self.parameters['filename']['value'], preload=True)
            
        if self.raw:
            chunk_samples = int(self.parameters['chunk_size']['value'] * self.raw.info['sfreq'])
            data = self.raw.get_data(start=self.current_position, 
                                   stop=self.current_position + chunk_samples)
            
            self.current_position += chunk_samples
            if self.current_position >= len(self.raw.times):
                if self.parameters['loop']['value']:
                    self.current_position = 0
                else:
                    self.stop_streaming()
                    
            self.outputs['data'] = data
            return {'data': data}
        
        return {'data': None}
        
    def required_inputs(self) -> List[str]:
        return []

class BufferNode(StreamNode):
    """Node for buffering streaming data."""
    
    def __init__(self):
        super().__init__("Buffer")
        self.inputs = {'data': None}
        self.outputs = {'buffered_data': None}
        self.parameters = {
            'buffer_size': {'type': 'float', 'value': 10.0},
            'overlap': {'type': 'float', 'value': 0.5}
        }
        self.buffer = None
        
    def process(self) -> Dict[str, Any]:
        """Process buffered data."""
        if not self.validate_inputs():
            return {'buffered_data': None}
            
        data = self.inputs['data']
        
        if self.buffer is None:
            self.buffer = data
        else:
            # Add new data to buffer
            self.buffer = np.concatenate([self.buffer, data], axis=1)
            
            # Remove old data if buffer is too large
            buffer_samples = int(self.parameters['buffer_size']['value'] * data.shape[1])
            if self.buffer.shape[1] > buffer_samples:
                overlap_samples = int(buffer_samples * self.parameters['overlap']['value'])
                self.buffer = self.buffer[:, -buffer_samples-overlap_samples:]
                
        self.outputs['buffered_data'] = self.buffer
        return {'buffered_data': self.buffer}
        
    def required_inputs(self) -> List[str]:
        return ['data']

class OnlineFilterNode(StreamNode):
    """Node for online filtering of streaming data."""
    
    def __init__(self):
        super().__init__("Online Filter")
        self.inputs = {'data': None}
        self.outputs = {'filtered_data': None}
        self.parameters = {
            'lowpass': {'type': 'float', 'value': 40.0},
            'highpass': {'type': 'float', 'value': 1.0},
            'order': {'type': 'int', 'value': 4}
        }
        self.filter_states = None
        
    def process(self) -> Dict[str, Any]:
        """Process streaming data with online filter."""
        if not self.validate_inputs():
            return {'filtered_data': None}
            
        data = self.inputs['data']
        
        # Design filter if not done
        if self.filter_states is None:
            self._design_filter(data.shape[0])
            
        # Apply filter
        filtered_data, self.filter_states = self._apply_filter(data)
        
        self.outputs['filtered_data'] = filtered_data
        return {'filtered_data': filtered_data}
        
    def _design_filter(self, n_channels: int):
        """Design online filter."""
        from scipy import signal
        
        nyq = self.sampling_rate / 2
        
        # Create filters
        if self.parameters['highpass']['value']:
            self.hp_b, self.hp_a = signal.butter(
                self.parameters['order']['value'],
                self.parameters['highpass']['value'] / nyq,
                btype='high'
            )
            self.hp_states = np.zeros((n_channels, 
                                     max(len(self.hp_a), len(self.hp_b)) - 1))
            
        if self.parameters['lowpass']['value']:
            self.lp_b, self.lp_a = signal.butter(
                self.parameters['order']['value'],
                self.parameters['lowpass']['value'] / nyq,
                btype='low'
            )
            self.lp_states = np.zeros((n_channels, 
                                     max(len(self.lp_a), len(self.lp_b)) - 1))
            
    def _apply_filter(self, data: np.ndarray) -> tuple:
        """Apply online filter to data chunk."""
        filtered_data = data.copy()
        
        if hasattr(self, 'hp_b'):
            filtered_data, self.hp_states = signal.lfilter(
                self.hp_b, self.hp_a, filtered_data, axis=1,
                zi=self.hp_states
            )
            
        if hasattr(self, 'lp_b'):
            filtered_data, self.lp_states = signal.lfilter(
                self.lp_b, self.lp_a, filtered_data, axis=1,
                zi=self.lp_states
            )
            
        return filtered_data, (self.hp_states if hasattr(self, 'hp_states') else None,
                             self.lp_states if hasattr(self, 'lp_states') else None)
        
    def required_inputs(self) -> List[str]:
        return ['data']