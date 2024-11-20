# tests/benchmarks/test_performance.py

import pytest
import time
import mne
import numpy as np
from pathlib import Path
from pymobi import PyMoBIConfig, PyMoBIData, create_default_pipeline

def generate_test_data(duration: float, n_channels: int, sfreq: float) -> mne.io.Raw:
    """Generate synthetic EEG data for testing."""
    n_samples = int(duration * sfreq)
    data = np.random.randn(n_channels, n_samples)
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info)

@pytest.mark.benchmark
class TestPipelinePerformance:
    """Test suite for pipeline performance benchmarks."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PyMoBIConfig(
            study_folder=Path("test_data"),
            resample_freq=250.0,
            channels_to_remove=[],
            ref_channel='FCz',
            asr_cutoff=20,
            use_asr=True,
            num_models=1,
            max_threads=4
        )
    
    @pytest.mark.parametrize("duration", [60, 300, 600])
    def test_processing_speed(self, duration, config, benchmark):
        """Benchmark processing speed for different data lengths."""
        # Generate test data
        raw = generate_test_data(
            duration=duration,
            n_channels=64,
            sfreq=1000.0
        )
        
        data = PyMoBIData(raw, subject_id=1)
        pipeline = create_default_pipeline(config)
        
        # Run benchmark
        result = benchmark(pipeline.run, data)
        
        # Verify result
        assert isinstance(result, PyMoBIData)
        assert result.mne_raw.info['sfreq'] == config.resample_freq
        
    @pytest.mark.parametrize("n_channels", [32, 64, 128])
    def test_channel_scaling(self, n_channels, config, benchmark):
        """Benchmark performance scaling with number of channels."""
        # Generate test data
        raw = generate_test_data(
            duration=60,
            n_channels=n_channels,
            sfreq=1000.0
        )
        
        data = PyMoBIData(raw, subject_id=1)
        pipeline = create_default_pipeline(config)
        
        # Run benchmark
        result = benchmark(pipeline.run, data)
        
        # Verify result
        assert isinstance(result, PyMoBIData)
        assert len(result.mne_raw.ch_names) == n_channels
        
    def test_memory_usage(self, config):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate large test data
        raw = generate_test_data(
            duration=300,
            n_channels=128,
            sfreq=1000.0
        )
        
        data = PyMoBIData(raw, subject_id=1)
        pipeline = create_default_pipeline(config)
        
        # Process data
        result = pipeline.run(data)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Check memory usage
        assert memory_increase < 2000  # Less than 2GB increase
        
    @pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
    def test_parallel_scaling(self, n_threads, config, benchmark):
        """Test processing speed with different numbers of threads."""
        config.max_threads = n_threads
        
        # Generate test data
        raw = generate_test_data(
            duration=60,
            n_channels=64,
            sfreq=1000.0
        )
        
        data = PyMoBIData(raw, subject_id=1)
        pipeline = create_default_pipeline(config)
        
        # Run benchmark
        result = benchmark(pipeline.run, data)
        
        # Verify result
        assert isinstance(result, PyMoBIData)
        
    def test_continuous_processing(self, config):
        """Test continuous processing of streaming data."""
        chunk_duration = 1.0  # 1 second chunks
        total_duration = 60.0  # 60 seconds total
        
        processing_times = []
        
        for i in range(int(total_duration / chunk_duration)):
            # Generate chunk
            raw = generate_test_data(
                duration=chunk_duration,
                n_channels=64,
                sfreq=1000.0
            )
            
            data = PyMoBIData(raw, subject_id=1)
            pipeline = create_default_pipeline(config)
            
            # Process chunk and measure time
            start_time = time.time()
            result = pipeline.run(data)
            processing_times.append(time.time() - start_time)
            
        # Calculate statistics
        mean_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        
        # Check processing speed consistency
        assert mean_time < chunk_duration  # Processing faster than real-time
        assert std_time < 0.1  # Consistent processing time