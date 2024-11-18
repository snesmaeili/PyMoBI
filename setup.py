from setuptools import setup, find_packages

setup(
    name='PyMoBI',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python library for mobile EEG data analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/PyMoBI',
    packages=find_packages(),
    install_requires=[
        'mne>=0.23.0',
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'matplotlib>=3.2.0',
        'pandas>=1.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
