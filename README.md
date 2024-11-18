PyMoBI: Python Mobile Brain Imaging Analysis
PyMoBI is a Python library designed for processing and analyzing mobile EEG (electroencephalography) data, with specialized support for motion-integrated brain imaging research. Built on top of MNE-Python, PyMoBI provides additional tools specifically designed for mobile brain imaging scenarios, overcoming limitations of existing tools and expanding capabilities to cover a variety of mobile tasks beyond gait analysis.

Table of Contents
Features
Installation
Quick Start
Documentation
Project Structure
Contributing
License
Contact
Features
Seamless Integration with MNE-Python: Utilize the powerful functionalities of MNE for EEG data handling.
Flexible Processing Pipeline: Customize the order of processing steps using object-oriented programming and class-based methods.
Motion Data Integration: Incorporate motion analysis using libraries like KielMAT or Python equivalents, enabling synchronization of EEG and motion data.
Advanced Artifact Detection and Removal: Implement methods like Zapline+ for line noise removal, ASR for artifact subspace reconstruction, and AMICA for advanced ICA decomposition.
Extensible and Modular Design: Easily add new processing methods and customize existing ones to fit your research needs.
Visualization Tools: Provide flexible visualization options for EEG and motion data.
Processing History Tracking: Maintain a detailed log of all processing steps for reproducibility.
Installation
Clone the repository and install the required dependencies:

bash
Copy code
git clone https://github.com/yourusername/PyMoBI.git
cd PyMoBI
pip install -r requirements.txt
Alternatively, you can install PyMoBI via pip (once it's available on PyPI):

bash
Copy code
pip install pymobi
Quick Start
Here's a basic example of how to use PyMoBI:

python
Copy code
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
Documentation
Comprehensive documentation is available in the docs/ directory or online at PyMoBI Documentation.

Project Structure
bash
Copy code
PyMoBI/
├── pymobi/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data.py          # Data container classes
│   │   └── config.py        # Configuration management
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── basic.py         # Basic MNE-based preprocessing
│   │   ├── zapline.py       # Implementation of Zapline+
│   │   ├── asr.py           # Implementation of ASR
│   │   ├── amica.py         # Implementation of AMICA
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── gait.py          # Gait event detection
│   │   └── sync.py          # EEG-motion synchronization
│   └── viz/
│       ├── __init__.py
│       ├── eeg_viz.py       # EEG visualization tools
│       └── motion_viz.py    # Motion data visualization
├── examples/
│   ├── basic_usage.py
│   └── gait_analysis.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_preprocessing.py
│   └── test_motion.py
├── docs/
│   ├── index.md
│   └── installation.md
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
└── .gitignore
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository on GitHub.

Clone your fork:

bash
Copy code
git clone https://github.com/yourusername/PyMoBI.git
Create a new branch:

bash
Copy code
git checkout -b feature/YourFeature
Commit your changes:

bash
Copy code
git commit -am 'Add a feature'
Push to the branch:

bash
Copy code
git push origin feature/YourFeature
Create a new Pull Request on GitHub.

Please read the CONTRIBUTING.md file for detailed guidelines on how to contribute to the project.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or suggestions, please contact:

Your Name
Email: your.email@example.com
GitHub: yourusername
