# pymobi/gui/nodes/visualization.py

from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mne
from ..core import BaseNode
import dearpygui.dearpygui as dpg

class SignalVisualizerNode(BaseNode):
    """Node for real-time signal visualization."""
    
    def __init__(self):
        super().__init__("Signal Visualizer")
        self.inputs = {'data': None}
        self.outputs = {'data': None}
        self.parameters = {
            'n_channels': {'type': 'int', 'value': 20},
            'time_window': {'type': 'float', 'value': 10.0},
            'scale': {'type': 'float', 'value': 50.0},
            'show_events': {'type': 'bool', 'value': True},
            'show_grid': {'type': 'bool', 'value': True},
            'plot_type': {
                'type': 'select',
                'value': 'line',
                'options': ['line', 'butterfly', 'image']
            }
        }
        self.fig = None
        self.ax = None
        
    def process(self) -> Dict[str, Any]:
        """Update visualization."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        
        # Update plot based on type
        if self.parameters['plot_type']['value'] == 'line':
            self._plot_line(data)
        elif self.parameters['plot_type']['value'] == 'butterfly':
            self._plot_butterfly(data)
        else:
            self._plot_image(data)
            
        self.outputs['data'] = data
        return {'data': data}
        
    def _plot_line(self, data: Any):
        """Plot channels as lines."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            
        self.ax.clear()
        
        # Get data segment
        n_samples = int(self.parameters['time_window']['value'] * 
                       data.mne_raw.info['sfreq'])
        segment = data.mne_raw.get_data(
            start=max(0, data.mne_raw.n_times - n_samples),
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
                    
        # Customize plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Channel')
        self.ax.set_title('EEG Signal')
        if self.parameters['show_grid']['value']:
            self.ax.grid(True)
            
        plt.draw()
        plt.pause(0.01)
        
    def _plot_butterfly(self, data: Any):
        """Plot channels overlaid (butterfly plot)."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            
        self.ax.clear()
        
        # Get data segment
        n_samples = int(self.parameters['time_window']['value'] * 
                       data.mne_raw.info['sfreq'])
        segment = data.mne_raw.get_data(
            start=max(0, data.mne_raw.n_times - n_samples),
            stop=data.mne_raw.n_times
        )
        
        # Plot channels
        times = np.linspace(
            0,
            self.parameters['time_window']['value'],
            segment.shape[1]
        )
        
        self.ax.plot(times, segment.T, alpha=0.5)
        
        # Customize plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('EEG Butterfly Plot')
        if self.parameters['show_grid']['value']:
            self.ax.grid(True)
            
        plt.draw()
        plt.pause(0.01)
        
    def _plot_image(self, data: Any):
        """Plot channels as image."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            
        self.ax.clear()
        
        # Get data segment
        n_samples = int(self.parameters['time_window']['value'] * 
                       data.mne_raw.info['sfreq'])
        segment = data.mne_raw.get_data(
            start=max(0, data.mne_raw.n_times - n_samples),
            stop=data.mne_raw.n_times
        )
        
        # Plot image
        self.ax.imshow(
            segment,
            aspect='auto',
            cmap='RdBu_r',
            extent=[0, self.parameters['time_window']['value'], 
                   segment.shape[0], 0]
        )
        
        # Customize plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Channel')
        self.ax.set_title('EEG Image Plot')
        
        plt.draw()
        plt.pause(0.01)
        
    def cleanup(self):
        """Clean up visualization resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
    def required_inputs(self) -> List[str]:
        return ['data']