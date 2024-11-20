# pymobi/gui/window.py

import dearpygui.dearpygui as dpg
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import mne
from ..core.config import PyMoBIConfig
from .nodes import NodeRegistry
from .pipeline import PipelineManager

class PyMoBIGUI:
    """Main GUI window for PyMoBI."""
    
    def __init__(self, config: PyMoBIConfig):
        """Initialize PyMoBI GUI."""
        self.config = config
        self.node_registry = NodeRegistry()
        self.pipeline_manager = PipelineManager()
        
        # Initialize GUI
        dpg.create_context()
        self.setup_theme()
        self.create_windows()
        
    def setup_theme(self):
        """Setup custom theme for better visualization."""
        with dpg.theme() as self.theme:
            with dpg.theme_component(dpg.mvAll):
                # Modern dark theme
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (60, 60, 60))
                
                # Node colors
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (40, 40, 40))
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (45, 45, 45))
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (100, 100, 100))
                
                # Connection colors
                dpg.add_theme_color(dpg.mvNodeCol_Link, (150, 150, 250))
                dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (200, 200, 255))
                
    def create_windows(self):
        """Create main windows for the pipeline builder."""
        # Main window with node editor
        with dpg.window(label="Pipeline Builder", tag="main_window", pos=(0, 0)):
            with dpg.node_editor(callback=self._on_connect, 
                               delink_callback=self._on_disconnect,
                               minimap=True,
                               minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                pass
                
        # Node menu
        with dpg.window(label="Nodes", tag="node_menu", 
                       width=250, height=600, pos=(0, 30)):
            self._create_node_menu()
            
        # Data viewer
        with dpg.window(label="Data Viewer", tag="data_viewer",
                       width=800, height=400, pos=(260, 30)):
            with dpg.plot(label="EEG Data", height=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
                dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude")
                
        # Properties window
        with dpg.window(label="Properties", tag="properties",
                       width=250, height=300, pos=(1070, 30)):
            pass
            
    def _create_node_menu(self):
        """Create node selection menu."""
        categories = {
            'Input': [
                {'name': 'File Loader', 'desc': 'Load EEG data from file'},
                {'name': 'Stream Input', 'desc': 'Real-time EEG input'}
            ],
            'Preprocessing': [
                {'name': 'Basic Preprocessing', 'desc': 'Filter, resample, ref'},
                {'name': 'ASR', 'desc': 'Artifact Subspace Reconstruction'},
                {'name': 'AMICA', 'desc': 'Advanced ICA decomposition'},
                {'name': 'Zapline+', 'desc': 'Line noise removal'}
            ],
            'Motion': [
                {'name': 'Motion Processing', 'desc': 'Process motion data'},
                {'name': 'Gait Events', 'desc': 'Detect gait events'}
            ],
            'Analysis': [
                {'name': 'Spectral', 'desc': 'Spectral analysis'},
                {'name': 'Connectivity', 'desc': 'Connectivity measures'}
            ],
            'Output': [
                {'name': 'Visualization', 'desc': 'Plot data'},
                {'name': 'Export', 'desc': 'Save processed data'}
            ]
        }
        
        for category, nodes in categories.items():
            with dpg.collapsing_header(label=category):
                for node in nodes:
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label=node['name'],
                            callback=lambda s, a, u: self._add_node(u),
                            user_data=node
                        )
                        dpg.add_text(node['desc'])