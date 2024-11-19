# pymobi/viz/reports.py

import mne
import numpy as np
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class ProcessingReport:
    """Generate comprehensive processing reports for PyMoBI analysis."""
    
    def __init__(self, config: PyMoBIConfig):
        """
        Initialize report generator with configuration.
        
        Parameters
        ----------
        config : PyMoBIConfig
            Configuration object containing report parameters
        """
        self.config = config
        
    def generate_report(self, data: PyMoBIData) -> str:
        """
        Generate complete processing report.
        
        Parameters
        ----------
        data : PyMoBIData
            Data container with processing history
            
        Returns
        -------
        str
            Path to generated report
        """
        # Create report directory
        report_path = self._get_report_path(data)
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report sections
        self._generate_processing_summary(data, report_path)
        self._generate_data_quality_report(data, report_path)
        self._generate_artifact_removal_report(data, report_path)
        self._generate_ica_report(data, report_path)
        
        # Combine into final HTML report
        report_file = self._create_html_report(data, report_path)
        
        return str(report_file)
        
    def _generate_processing_summary(self, data: PyMoBIData, report_path: Path):
        """Generate processing steps summary."""
        summary = {
            'subject_id': data.subject_id,
            'processing_date': datetime.now().isoformat(),
            'sampling_rate': data.mne_raw.info['sfreq'],
            'n_channels': len(data.mne_raw.ch_names),
            'duration': data.mne_raw.times[-1],
            'processing_steps': data.processing_history
        }
        
        # Save summary
        with open(report_path / 'processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _generate_data_quality_report(self, data: PyMoBIData, report_path: Path):
        """Generate data quality metrics and plots."""
        fig = plt.figure(figsize=(15, 10))
        
        # Plot data overview
        ax1 = plt.subplot(211)
        data.mne_raw.plot(duration=30, n_channels=20, ax=ax1)
        ax1.set_title('Data Overview (First 30s)')
        
        # Plot channel variance
        ax2 = plt.subplot(212)
        channel_var = np.var(data.mne_raw.get_data(), axis=1)
        ax2.bar(range(len(channel_var)), channel_var)
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Variance')
        ax2.set_title('Channel Variance Distribution')
        
        plt.tight_layout()
        plt.savefig(report_path / 'data_quality.png')
        plt.close()
        
    def _generate_artifact_removal_report(self, data: PyMoBIData, report_path: Path):
        """Generate artifact removal summary and plots."""
        if not hasattr(data, 'bad_channels'):
            return
            
        # Create summary
        artifact_summary = {
            'bad_channels': data.bad_channels,
            'interpolated_channels': data.interpolated_channels,
            'n_bad_channels': len(data.bad_channels),
            'percent_bad_channels': len(data.bad_channels) / len(data.mne_raw.ch_names) * 100
        }
        
        # Save summary
        with open(report_path / 'artifact_summary.json', 'w') as f:
            json.dump(artifact_summary, f, indent=2)
            
        # Plot bad channel locations
        if data.mne_raw.info['dig'] is not None:
            fig = plt.figure(figsize=(10, 10))
            mne.viz.plot_sensors(data.mne_raw.info, 
                               kind='topomap', 
                               ch_type='eeg',
                               show_names=True)
            plt.savefig(report_path / 'bad_channels_location.png')
            plt.close()
            
    def _generate_ica_report(self, data: PyMoBIData, report_path: Path):
        """Generate ICA component analysis report."""
        if data.ica is None:
            return
            
        # Create ICA summary
        ica_summary = {
            'n_components': data.ica.n_components_,
            'n_excluded': len(data.ica_excluded),
            'excluded_components': data.ica_excluded
        }
        
        # Save summary
        with open(report_path / 'ica_summary.json', 'w') as f:
            json.dump(ica_summary, f, indent=2)
            
        # Plot components
        fig = data.ica.plot_components(show=False)
        plt.savefig(report_path / 'ica_components.png')
        plt.close()
        
        # Plot component properties if ICLabel scores available
        if data.iclabel_scores is not None:
            fig = plt.figure(figsize=(15, 5))
            plt.bar(range(len(data.iclabel_scores)), 
                   data.iclabel_scores.max(axis=1))
            plt.xlabel('Component Index')
            plt.ylabel('Max ICLabel Score')
            plt.title('ICLabel Component Classification')
            plt.savefig(report_path / 'iclabel_scores.png')
            plt.close()
            
    def _create_html_report(self, data: PyMoBIData, report_path: Path) -> Path:
        """Create final HTML report combining all sections."""
        report_file = report_path / 'processing_report.html'
        
        # Load summaries
        with open(report_path / 'processing_summary.json', 'r') as f:
            processing_summary = json.load(f)
            
        html_content = f"""
        <html>
        <head>
            <title>PyMoBI Processing Report - Subject {data.subject_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>PyMoBI Processing Report</h1>
            <div class="section">
                <h2>Processing Summary</h2>
                <table>
                    <tr><th>Subject ID</th><td>{processing_summary['subject_id']}</td></tr>
                    <tr><th>Processing Date</th><td>{processing_summary['processing_date']}</td></tr>
                    <tr><th>Sampling Rate</th><td>{processing_summary['sampling_rate']} Hz</td></tr>
                    <tr><th>Number of Channels</th><td>{processing_summary['n_channels']}</td></tr>
                    <tr><th>Duration</th><td>{processing_summary['duration']:.2f} s</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Data Quality</h2>
                <img src="data_quality.png" alt="Data Quality Overview">
            </div>
        """
        
        # Add artifact section if available
        if (report_path / 'artifact_summary.json').exists():
            with open(report_path / 'artifact_summary.json', 'r') as f:
                artifact_summary = json.load(f)
                
            html_content += f"""
            <div class="section">
                <h2>Artifact Removal</h2>
                <table>
                    <tr><th>Bad Channels</th><td>{', '.join(artifact_summary['bad_channels'])}</td></tr>
                    <tr><th>Number of Bad Channels</th><td>{artifact_summary['n_bad_channels']}</td></tr>
                    <tr><th>Percent Bad Channels</th><td>{artifact_summary['percent_bad_channels']:.2f}%</td></tr>
                </table>
                <img src="bad_channels_location.png" alt="Bad Channel Locations">
            </div>
            """
            
        # Add ICA section if available
        if (report_path / 'ica_summary.json').exists():
            with open(report_path / 'ica_summary.json', 'r') as f:
                ica_summary = json.load(f)
                
            html_content += f"""
            <div class="section">
                <h2>ICA Analysis</h2>
                <table>
                    <tr><th>Number of Components</th><td>{ica_summary['n_components']}</td></tr>
                    <tr><th>Excluded Components</th><td>{', '.join(map(str, ica_summary['excluded_components']))}</td></tr>
                </table>
                <img src="ica_components.png" alt="ICA Components">
                <img src="iclabel_scores.png" alt="ICLabel Scores">
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(report_file, 'w') as f:
            f.write(html_content)
            
        return report_file
        
    def _get_report_path(self, data: PyMoBIData) -> Path:
        """Get report directory path."""
        base_path = Path(self.config.study_folder) / 'reports'
        if data.subject_id is not None:
            return base_path / f"sub-{data.subject_id}"
        return base_path / "unknown_subject"