"""
Neuroscience R&D Assistant for GPT-based LLM
Main module for coordinating neuroscience research workflows
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for neuroscience experiments"""
    experiment_id: str
    title: str
    description: str
    organism: str  # e.g., "Mus musculus", "Homo sapiens"
    brain_region: str
    technique: str  # e.g., "fMRI", "electrophysiology", "calcium imaging"
    timestamp: str
    principal_investigator: str
    protocol_url: Optional[str] = None
    notes: Optional[str] = None


class NeuroscienceRnDAssistant:
    """Main class for neuroscience R&D assistance"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize the neuroscience R&D assistant
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.experiments: Dict[str, ExperimentMetadata] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
    def create_experiment(
        self,
        experiment_id: str,
        title: str,
        organism: str,
        brain_region: str,
        technique: str,
        description: str = "",
        pi: str = "",
        protocol_url: str = None,
        notes: str = None
    ) -> ExperimentMetadata:
        """Create a new experiment record"""
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            title=title,
            organism=organism,
            brain_region=brain_region,
            technique=technique,
            description=description,
            principal_investigator=pi,
            timestamp=datetime.now().isoformat(),
            protocol_url=protocol_url,
            notes=notes
        )
        self.experiments[experiment_id] = metadata
        logger.info(f"Created experiment: {experiment_id}")
        return metadata
    
    def save_experiment_metadata(self, output_path: str) -> None:
        """Save all experiment metadata to JSON file"""
        data = {
            exp_id: asdict(metadata)
            for exp_id, metadata in self.experiments.items()
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved experiment metadata to {output_path}")


class DataHandler:
    """Handle various neuroscience data formats"""
    
    @staticmethod
    def load_timeseries(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load electrophysiology or calcium imaging timeseries data
        
        Args:
            filepath: Path to data file (CSV, NPZ, or HDF5)
            
        Returns:
            Tuple of (data_array, metadata_dict)
        """
        if filepath.endswith('.csv'):
            data = np.loadtxt(filepath, delimiter=',')
            return data, {'format': 'csv', 'shape': data.shape}
        elif filepath.endswith('.npz'):
            loaded = np.load(filepath)
            data = loaded['data']
            metadata = {k: v.item() if isinstance(v, np.ndarray) else v 
                       for k, v in loaded.items() if k != 'data'}
            return data, metadata
        else:
            logger.warning(f"Unsupported file format: {filepath}")
            return None, {}
    
    @staticmethod
    def save_timeseries(data: np.ndarray, filepath: str, metadata: Dict = None) -> None:
        """Save timeseries data with metadata"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        if filepath.endswith('.npz'):
            save_dict = {'data': data}
            if metadata:
                save_dict.update(metadata)
            np.savez(filepath, **save_dict)
        else:
            np.savetxt(filepath, data, delimiter=',')
        logger.info(f"Saved data to {filepath}")


class AnalysisTools:
    """Neuroscience analysis utilities"""
    
    @staticmethod
    def compute_firing_rate(spike_times: np.ndarray, window_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute firing rate from spike times using sliding window
        
        Args:
            spike_times: Array of spike times in seconds
            window_size: Window size in seconds
            
        Returns:
            Tuple of (firing_rates, time_bins)
        """
        if len(spike_times) == 0:
            return np.array([]), np.array([])
        
        max_time = spike_times.max()
        time_bins = np.arange(0, max_time, window_size)
        firing_rates = []
        
        for t_start in time_bins:
            t_end = t_start + window_size
            spikes_in_window = np.sum((spike_times >= t_start) & (spike_times < t_end))
            firing_rate = spikes_in_window / window_size  # Hz
            firing_rates.append(firing_rate)
        
        return np.array(firing_rates), time_bins
    
    @staticmethod
    def compute_cross_correlation(signal1: np.ndarray, signal2: np.ndarray, max_lag: int = 100) -> np.ndarray:
        """Compute cross-correlation between two signals"""
        lags = np.arange(-max_lag, max_lag + 1)
        correlations = []
        
        signal1_normalized = (signal1 - np.mean(signal1)) / np.std(signal1)
        signal2_normalized = (signal2 - np.mean(signal2)) / np.std(signal2)
        
        for lag in lags:
            if lag < 0:
                corr = np.mean(signal1_normalized[:lag] * signal2_normalized[-lag:])
            elif lag > 0:
                corr = np.mean(signal1_normalized[lag:] * signal2_normalized[:-lag])
            else:
                corr = np.mean(signal1_normalized * signal2_normalized)
            correlations.append(corr)
        
        return np.array(correlations)
    
    @staticmethod
    def compute_raster_metrics(spike_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compute population-level metrics from spike raster
        
        Args:
            spike_matrix: Shape (n_neurons, n_timepoints), 1=spike, 0=no spike
            
        Returns:
            Dictionary with population metrics
        """
        n_neurons, n_timepoints = spike_matrix.shape
        
        return {
            'total_spikes': int(np.sum(spike_matrix)),
            'mean_firing_rate': float(np.sum(spike_matrix) / (n_neurons * n_timepoints)),
            'spikes_per_neuron': np.sum(spike_matrix, axis=1).tolist(),
            'population_synchrony': float(np.mean(np.corrcoef(spike_matrix))),
            'n_neurons': n_neurons,
            'n_timepoints': n_timepoints,
        }


class ExperimentDesigner:
    """Helper for designing neuroscience experiments"""
    
    @staticmethod
    def suggest_sample_size(
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: str = "t-test"
    ) -> Dict[str, Any]:
        """
        Suggest sample size for statistical test
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Statistical power (1 - beta)
            test_type: Type of statistical test
            
        Returns:
            Sample size recommendation and rationale
        """
        # Simplified power analysis (in production, use scipy.stats)
        # For two-sample t-test: n = 2 * (z_alpha + z_beta)^2 / d^2
        
        z_alpha = 1.96 if alpha == 0.05 else 2.576  # One-tailed approximation
        z_beta = 0.84 if power == 0.8 else 1.28  # One-tailed approximation
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return {
            'test_type': test_type,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'suggested_n_per_group': int(np.ceil(n_per_group)),
            'total_n': int(np.ceil(2 * n_per_group)),
            'note': 'Use specialized software (G*Power, statsmodels) for precise calculations'
        }
    
    @staticmethod
    def create_experimental_plan(
        hypothesis: str,
        organism: str,
        brain_region: str,
        expected_duration_hours: float,
        n_animals: int
    ) -> Dict[str, Any]:
        """Create a basic experimental plan template"""
        return {
            'hypothesis': hypothesis,
            'organism': organism,
            'brain_region': brain_region,
            'expected_duration_hours': expected_duration_hours,
            'n_animals': n_animals,
            'timeline': {
                'preparation': 1.0,
                'recording': expected_duration_hours - 2.0,
                'recovery': 1.0,
            },
            'data_requirements': {
                'sampling_rate_hz': 30000,  # Example for electrophysiology
                'expected_file_size_gb': None,  # To be calculated
            },
            'analysis_steps': [
                'Quality control and preprocessing',
                'Feature extraction',
                'Statistical analysis',
                'Visualization and interpretation'
            ]
        }


class KnowledgeBase:
    """Store and retrieve neuroscience research knowledge"""
    
    def __init__(self):
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.tags: Dict[str, List[str]] = {}
    
    def add_entry(self, key: str, content: str, tags: List[str] = None, metadata: Dict = None) -> None:
        """Add an entry to the knowledge base"""
        self.entries[key] = {
            'content': content,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        if tags:
            self.tags[key] = tags
        logger.info(f"Added knowledge base entry: {key}")
    
    def search_by_tag(self, tag: str) -> List[str]:
        """Find all entries with a given tag"""
        return [key for key, tag_list in self.tags.items() if tag in tag_list]
    
    def get_entry(self, key: str) -> Optional[Dict]:
        """Retrieve an entry from the knowledge base"""
        return self.entries.get(key)
    
    def export_knowledge_base(self, filepath: str) -> None:
        """Export knowledge base to JSON"""
        export_data = {
            'entries': self.entries,
            'tags': self.tags,
            'exported_at': datetime.now().isoformat()
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Exported knowledge base to {filepath}")


def create_llm_context(
    assistant: NeuroscienceRnDAssistant,
    experiment_id: str,
    analysis_type: str
) -> str:
    """
    Create a context string for the LLM based on experiment and analysis type
    
    Args:
        assistant: NeuroscienceRnDAssistant instance
        experiment_id: ID of the experiment
        analysis_type: Type of analysis to perform
        
    Returns:
        Context string for the LLM prompt
    """
    if experiment_id not in assistant.experiments:
        return f"Error: Experiment {experiment_id} not found"
    
    exp = assistant.experiments[experiment_id]
    
    context = f"""
NEUROSCIENCE RESEARCH CONTEXT
=============================
Experiment ID: {exp.experiment_id}
Title: {exp.title}
Organism: {exp.organism}
Brain Region: {exp.brain_region}
Technique: {exp.technique}
Description: {exp.description}
Principal Investigator: {exp.principal_investigator}
Timestamp: {exp.timestamp}

TASK: {analysis_type}

Please provide:
1. Relevant analysis methods for this experimental setup
2. Expected outcomes and interpretation
3. Potential confounds and controls
4. Suggestions for data quality metrics
5. Relevant literature references (if available)
"""
    return context


if __name__ == "__main__":
    # Example usage
    assistant = NeuroscienceRnDAssistant()
    
    # Create an experiment
    exp = assistant.create_experiment(
        experiment_id="EXP_001",
        title="Single-unit recording in V1",
        organism="Mus musculus",
        brain_region="Visual Cortex (V1)",
        technique="Electrophysiology",
        description="Recording from layer 2/3 neurons during visual stimulus presentation",
        pi="Dr. Smith",
        notes="Using 32-channel probe"
    )
    
    # Save metadata
    assistant.save_experiment_metadata("outputs/experiment_metadata.json")
    
    # Example analysis
    print("\nSample firing rate calculation:")
    spike_times = np.array([0.05, 0.12, 0.15, 0.23, 0.31])
    firing_rates, bins = AnalysisTools.compute_firing_rate(spike_times)
    print(f"Firing rates (Hz): {firing_rates}")
    print(f"Time bins (s): {bins}")
    
    # Example experiment design
    print("\nSample size recommendation:")
    rec = ExperimentDesigner.suggest_sample_size(effect_size=0.8)
    print(f"Suggested n per group: {rec['suggested_n_per_group']}")
    print(f"Total n: {rec['total_n']}")
    
    # Create LLM context
    context = create_llm_context(assistant, "EXP_001", "neural_encoding_analysis")
    print("\nLLM Context for analysis:")
    print(context)
