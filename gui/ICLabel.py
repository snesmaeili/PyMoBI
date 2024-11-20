# pymobi/gui/nodes/iclabel.py

from typing import Dict, Any, List
import numpy as np
import mne
from ..core import BaseNode

class ICLabelNode(BaseNode):
    """Node for ICLabel classification and cleaning."""
    
    def __init__(self):
        super().__init__("ICLabel")
        self.inputs = {'data': None}
        self.outputs = {'data': None}
        self.parameters = {
            # ICLabel parameters from BeMoBIL
            'classifier': {
                'type': 'select',
                'value': 'lite',
                'options': ['lite', 'default']
            },
            'threshold': {'type': 'float', 'value': -1},
            'classes': {
                'type': 'multiselect',
                'value': [1],
                'options': [
                    {'value': 1, 'label': 'Brain'},
                    {'value': 2, 'label': 'Muscle'},
                    {'value': 3, 'label': 'Eye'},
                    {'value': 4, 'label': 'Heart'},
                    {'value': 5, 'label': 'Line Noise'},
                    {'value': 6, 'label': 'Channel Noise'},
                    {'value': 7, 'label': 'Other'}
                ]
            },
            'plot_removed': {'type': 'bool', 'value': True}
        }
        
    def process(self) -> Dict[str, Any]:
        """Run ICLabel classification and cleaning."""
        if not self.validate_inputs():
            raise ValueError("Missing required inputs")
            
        data = self.inputs['data']
        
        if data.ica is None:
            raise ValueError("No ICA solution found. Run ICA first.")
            
        try:
            # Run ICLabel
            data = self._run_iclabel(data)
            
            # Clean data based on ICLabel results
            data = self._clean_with_iclabel(data)
            
            # Plot if requested
            if self.parameters['plot_removed']['value']:
                self._plot_removed_components(data)
                
        except Exception as e:
            print(f"Error during ICLabel processing: {str(e)}")
            raise
            
        self.outputs['data'] = data
        return {'data': data}
        
    def _run_iclabel(self, data: Any) -> Any:
        """Run ICLabel classification."""
        # Compute ICLabel scores
        data.iclabel_scores = self._compute_iclabel_scores(
            data.ica,
            classifier=self.parameters['classifier']['value']
        )
        
        # Store classification info
        data.metadata['iclabel'] = {
            'classifier': self.parameters['classifier']['value'],
            'classes': self.parameters['classes']['value'],
            'threshold': self.parameters['threshold']['value']
        }
        
        return data
        
    def _clean_with_iclabel(self, data: Any) -> Any:
        """Clean data using ICLabel results."""
        # Get components to remove
        components_to_remove = self._get_components_to_remove(
            data.iclabel_scores,
            self.parameters['classes']['value'],
            self.parameters['threshold']['value']
        )
        
        # Store removed components
        data.metadata['removed_components'] = components_to_remove
        
        # Apply cleaning
        data.ica.exclude = components_to_remove
        data.mne_raw = data.ica.apply(data.mne_raw)
        
        return data
        
    def _compute_iclabel_scores(self, ica: mne.preprocessing.ICA, 
                              classifier: str) -> np.ndarray:
        """Compute ICLabel scores."""
        # This would integrate with ICLabel implementation
        # For now, return random scores for testing
        n_components = ica.n_components_
        n_classes = 7  # Number of ICLabel classes
        
        return np.random.rand(n_components, n_classes)
        
    def _get_components_to_remove(self, scores: np.ndarray, 
                                classes: List[int],
                                threshold: float) -> List[int]:
        """Get components to remove based on ICLabel scores."""
        if threshold == -1:
            # Use popularity classifier (highest probability class)
            classifications = np.argmax(scores, axis=1) + 1
            return [i for i, c in enumerate(classifications) 
                   if c not in classes]
        else:
            # Use threshold
            class_scores = np.sum(scores[:, [c-1 for c in classes]], axis=1)
            return [i for i, score in enumerate(class_scores) 
                   if score < threshold]
            
    def _plot_removed_components(self, data: Any):
        """Plot removed components."""
        # Implementation of component visualization
        pass
        
    def required_inputs(self) -> List[str]:
        return ['data']