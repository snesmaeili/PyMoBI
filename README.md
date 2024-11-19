# PyMoBI: Python Mobile Brain Imaging Analysis

PyMoBI is a Python library designed for processing and analyzing mobile EEG (electroencephalography) data, with specialized support for motion-integrated brain imaging research. Built on top of MNE-Python, PyMoBI provides additional tools specifically designed for mobile brain imaging scenarios, overcoming limitations of existing tools and expanding capabilities to cover a variety of mobile tasks beyond gait analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Seamless Integration with MNE-Python**: Utilize the powerful functionalities of MNE for EEG data handling.
- **Flexible Processing Pipeline**: Customize the order of processing steps using object-oriented programming and class-based methods.
- **Motion Data Integration**: Incorporate motion analysis using libraries like KielMAT or Python equivalents, enabling synchronization of EEG and motion data.
- **Advanced Artifact Detection and Removal**: Implement methods like Zapline+ for line noise removal, ASR for artifact subspace reconstruction, and AMICA for advanced ICA decomposition.
- **Extensible and Modular Design**: Easily add new processing methods and customize existing ones to fit your research needs.
- **Visualization Tools**: Provide flexible visualization options for EEG and motion data.
- **Processing History Tracking**: Maintain a detailed log of all processing steps for reproducibility.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/PyMoBI.git
cd PyMoBI
pip install -r requirements.txt
```

Alternatively, you can install PyMoBI via pip (once it's available on PyPI):

```bash
pip install pymobi
```

## Quick Start

Here's a basic example of how to use PyMoBI:

```python
import mne
from pymobi.core.data import PyMoBIData
from pymobi.core.config import PyMoBIConfig
from pymobi.preprocessing.pipeline import create_mobile_pipeline

# Load your raw EEG data
raw_eeg = mne.io.read_raw_fif('your_eeg_data.fif', preload=True)

# Load motion data if available
# Replace 'load_motion_data' with your actual motion data loading function
motion_data = load_motion_data('your_motion_data.c3d')  

# Initialize PyMoBI data object
data = PyMoBIData(raw_eeg, motion_data)

# Set up configuration
config = PyMoBIConfig(
    mne_preprocessing={
        'l_freq': 1.0,
        'h_freq': 40.0,
        'resample_freq': 250
    },
    motion_preprocessing={
        'detect_gait_events': True
    },
    custom_processing={
        'use_zapline': True,
        'use_asr': True,
        'use_amica': True
    },
    visualization={
        'plot_type': 'standard'
    }
)

# Create processing pipeline
pipeline = create_mobile_pipeline(config)

# Run the processing pipeline
for step in pipeline:
    data = step.run(data, config)

# Visualize the results
data.mne_raw.plot(n_channels=30, block=True)

# Save the processed data
data.mne_raw.save('processed_data.fif', overwrite=True)
```

## Documentation

Comprehensive documentation is available in the `docs/` directory or online at [PyMoBI Documentation](#).

## Project Structure

```bash
PyMoBI/
├── pymobi/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── data.py           # Data container
│   │   ├── logger.py         # Logging system
│   │   └── pipeline.py       # Pipeline orchestration
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers.py        # Data readers (BIDS, XDF, etc.)
│   │   └── writers.py        # BIDS export
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── basic.py          # Basic preprocessing
│   │   ├── artifacts.py      # Artifact removal (ASR, etc.)
│   │   └── ica.py           # ICA processing (AMICA, etc.)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── spectral.py       # Spectral analysis
│   │   └── connectivity.py   # Connectivity analysis
│   └── viz/
│       ├── __init__.py
│       ├── signals.py        # Signal visualization
│       ├── topography.py     # Topographic plots
│       └── reports.py        # Processing reports
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository on GitHub.

2. Clone your fork:

   ```bash
   git clone https://github.com/yourusername/PyMoBI.git
   ```

3. Create a new branch:

   ```bash
   git checkout -b feature/YourFeature
   ```

4. Commit your changes:

   ```bash
   git commit -am 'Add a feature'
   ```

5. Push to the branch:

   ```bash
   git push origin feature/YourFeature
   ```

6. Create a new Pull Request on GitHub.

Please read the `CONTRIBUTING.md` file for detailed guidelines on how to contribute to the project.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For questions or suggestions, please contact:\
Email: [sina.esmeili@umontreal.ca](mailto\:sina.esmeili@umontreal.ca)

