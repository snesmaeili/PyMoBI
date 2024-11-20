# pymobi/gui/nodes/signal.py

from typing import Dict, Any, List
import numpy as np
import mne
from ..core import BaseNode
import matplotlib.pyplot as plt

class SignalProcessingNode(BaseNode):
    """Node for basic signal processing operations."""
    
    def __init__(self):
        super().__init__("Signal Processing")
        self.inputs = {'data': None}
        self.outputs = {'data': None}
        self.parameters = {
            'lowpass': {'type': 'float', 'value': 40.0},
            'highpass': {'type': 'float', 'value': 1.0},
            'notch': {'type': 'float', 'value': 50.0},
            'resample': {'type': 'float', 'value': 250.0},
            'apply_car': {'type': 'bool', 'value': True}
        }
        
    def process(self) -> Dict[str, Any]:
        """Process the signal."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        
        # Apply filters
        if self.parameters['lowpass']['value']:
            data.mne_raw.filter(
                l_freq=None,
                h_freq=self.parameters['lowpass']['value']
            )
            
        if self.parameters['highpass']['value']:
            data.mne_raw.filter(
                l_freq=self.parameters['highpass']['value'],
                h_freq=None
            )
            
        # Apply notch filter
        if self.parameters['notch']['value']:
            data.mne_raw.notch_filter(
                self.parameters['notch']['value']
            )
            
        # Resample
        if self.parameters['resample']['value']:
            data.mne_raw.resample(
                self.parameters['resample']['value']
            )
            
        # Apply CAR
        if self.parameters['apply_car']['value']:
            data.mne_raw.set_eeg_reference('average')
            
        self.outputs['data'] = data
        return {'data': data}
        
    def required_inputs(self) -> List[str]:
        return ['data']

class VisualizationNode(BaseNode):
    """Node for real-time signal visualization."""
    
    def __init__(self):
        super().__init__("Visualization")
        self.inputs = {'data': None}
        self.outputs = {'data': None}
        self.parameters = {
            'n_channels': {'type': 'int', 'value': 20},
            'time_window': {'type': 'float', 'value': 10.0},
            'scale': {'type': 'float', 'value': 50.0},
            'show_events': {'type': 'bool', 'value': True},
            'show_filters': {'type': 'bool', 'value': True}
        }
        self.fig = None
        self.ax = None
        
    def process(self) -> Dict[str, Any]:
        """Update visualization."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        
        # Create or update plot
        self._update_plot(data)
        
        self.outputs['data'] = data
        return {'data': data}
        
    def _update_plot(self, data: Any):
        """Update the visualization plot."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            plt.ion()
            
        self.ax.clear()
        
        # Get data segment
        start_idx = max(0, data.mne_raw.n_times - 
                       int(self.parameters['time_window']['value'] * 
                           data.mne_raw.info['sfreq']))
        segment = data.mne_raw.get_data(
            start=start_idx,
            stop=data.mne_raw.n_times
        )
        
        # Plot channels
        times = np.linspace(
            0,
            self.parameters['time_window']['value'],
            segment.shape[1]
        )
        
        for i in range(min(self.parameters['n_channels']['value'], 
                          segment.shape[0])):
            self.ax.plot(
                times,
                segment[i] + i * self.parameters['scale']['value'],
                label=data.mne_raw.ch_names[i]
            )
            
        # Add events if requested
        if self.parameters['show_events']['value']:
            for event in data.mne_raw.annotations:
                if event['onset'] >= times[0] and event['onset'] <= times[-1]:
                    self.ax.axvline(
                        event['onset'],
                        color='r',
                        linestyle='--',
                        alpha=0.5
                    )
                    
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Channel')
        self.ax.set_title('EEG Signal Visualization')
        self.ax.grid(True)
        
        plt.draw()
        plt.pause(0.01)
        
    def required_inputs(self) -> List[str]:
        return ['data']
        
    def cleanup(self):
        """Clean up visualization resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None