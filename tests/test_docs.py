# tests/test_docs.py

import pytest
import doctest
import inspect
from pathlib import Path
import pymobi
from pymobi.core import PyMoBIConfig, PyMoBIData, PyMoBILogger
from pymobi.preprocessing import (
    BasicPreprocessing,
    ArtifactRemoval,
    PyASR,
    ICAProcessor,
    PreprocessingPipeline
)

class TestDocumentation:
    """Test suite for documentation and API consistency."""
    
    def collect_public_modules(self):
        """Collect all public modules in PyMoBI."""
        modules = []
        root_dir = Path(pymobi.__file__).parent
        
        for py_file in root_dir.rglob("*.py"):
            if py_file.stem.startswith("_"):
                continue
                
            module_path = str(py_file.relative_to(root_dir.parent)).replace("/", ".")
            module_name = module_path[:-3]  # Remove .py extension
            
            try:
                module = __import__(module_name, fromlist=[""])
                modules.append(module)
            except ImportError:
                continue
                
        return modules
        
    def test_docstrings_exist(self):
        """Test that all public methods have docstrings."""
        modules = self.collect_public_modules()
        
        for module in modules:
            for name, obj in inspect.getmembers(module):
                if name.startswith("_"):
                    continue
                    
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    doc = inspect.getdoc(obj)
                    assert doc is not None, f"Missing docstring for {module.__name__}.{name}"
                    assert len(doc.strip()) > 0, f"Empty docstring for {module.__name__}.{name}"
                    
    def test_docstring_format(self):
        """Test that docstrings follow NumPy format."""
        modules = self.collect_public_modules()
        
        for module in modules:
            for name, obj in inspect.getmembers(module):
                if name.startswith("_"):
                    continue
                    
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    doc = inspect.getdoc(obj)
                    if doc:
                        # Check for Parameters section
                        if "Parameters" in doc:
                            assert "----------" in doc, f"Invalid Parameters format in {module.__name__}.{name}"
                            
                        # Check for Returns section
                        if "Returns" in doc:
                            assert "-------" in doc, f"Invalid Returns format in {module.__name__}.{name}"
                            
    def test_doctest_examples(self):
        """Test that all doctest examples work."""
        modules = self.collect_public_modules()
        
        for module in modules:
            doctest.testmod(module, raise_on_error=True)
            
    def test_api_consistency(self):
        """Test API consistency across modules."""
        # Test core API
        assert hasattr(PyMoBIConfig, 'validate')
        assert hasattr(PyMoBIConfig, 'create_folders')
        
        assert hasattr(PyMoBIData, 'add_processing_step')
        assert hasattr(PyMoBIData, 'save')
        assert hasattr(PyMoBIData, 'load')
        
        # Test preprocessing API
        assert hasattr(BasicPreprocessing, 'run')
        assert hasattr(ArtifactRemoval, 'run')
        assert hasattr(PyASR, 'run')
        assert hasattr(ICAProcessor, 'run')
        
        # Test pipeline API
        assert hasattr(PreprocessingPipeline, 'add_step')
        assert hasattr(PreprocessingPipeline, 'run')
        
    def test_error_messages(self):
        """Test that error messages are informative."""
        config = PyMoBIConfig(study_folder="nonexistent")
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "folder does not exist" in str(exc_info.value)
        
    def test_deprecation_warnings(self):
        """Test that deprecated features raise warnings."""
        # Example of checking deprecation warnings
        with pytest.warns(DeprecationWarning):
            # Call deprecated method or use deprecated parameter
            pass
            
    def test_config_validation(self):
        """Test configuration validation rules."""
        invalid_configs = [
            {'resample_freq': -1},
            {'chan_detected_fraction_threshold': 1.5},
            {'asr_cutoff': 'invalid'},
            {'iclabel_threshold': 2.0}
        ]
        
        for invalid_params in invalid_configs:
            config = PyMoBIConfig(
                study_folder="test_data",
                **invalid_params
            )
            with pytest.raises(ValueError):
                config.validate()