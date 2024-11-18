# pymobi/core/pipeline.py

from abc import ABC, abstractmethod

class ProcessingStep(ABC):
    """Abstract base class for all processing steps."""
    
    @abstractmethod
    def run(self, data, config):
        pass

class PreprocessingPipeline:
    """Pipeline to manage and run preprocessing steps."""
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, step: ProcessingStep):
        """Add a processing step to the pipeline."""
        self.steps.append(step)
    
    def run(self, data, config):
        """Run all steps in the pipeline."""
        for step in self.steps:
            data = step.run(data, config)
        return data