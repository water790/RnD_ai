"""
Data visualization utilities for neuroscience research
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class NeuroscienceVisualizations:
    """Utilities for creating neuroscience-specific visualizations"""
    
    @staticmethod
    def spike_raster_data(
        spike_times_list: List[np.ndarray],
        trial_starts: np.ndarray = None,
        stimulus_window: Tuple[float, float] = None
    ) -> Dict:
        """
        Prepare spike raster plot data
        
        Args:
            spike_times_list: List of spike time arrays, one per neuron/unit
            trial_starts: Start times of trials (for aligning trials)
            stimulus_window: (start, end) of stimulus window relative to trial start
            
        Returns:
            Dictionary with data formatted for raster plot
        """
        raster_data = {
            'neurons': [],
            'spike_times': [],
            'spike_counts': [],
            'trial_info': None
        }
        
        for neuron_idx, spike_times in enumerate(spike_times_list):
            raster_data['neurons'].append(neuron_idx)
            raster_data['spike_times'].append(spike_times.tolist() if isinstance(spike_times, np.ndarray) else spike_times)
            raster_data['spike_counts'].append(len(spike_times))
        
        if trial_starts is not None:
            raster_data['trial_info'] = {
                'trial_starts': trial_starts.tolist() if isinstance(trial_starts, np.ndarray) else trial_starts,
                'stimulus_window': stimulus_window
            }
        
        return raster_data
    
    @staticmethod
    def heatmap_data(
        data: np.ndarray,
        rows_label: str = "Neurons",
        cols_label: str = "Time",
        normalize: bool = True
    ) -> Dict:
        """
        Prepare data for heatmap visualization (e.g., neural responses)
        
        Args:
            data: 2D array (rows x columns)
            rows_label: Label for rows
            cols_label: Label for columns
            normalize: Whether to normalize between 0 and 1
            
        Returns:
            Dictionary with heatmap data
        """
        if normalize:
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data_normalized = (data - data_min) / (data_max - data_min)
            else:
                data_normalized = data
        else:
            data_normalized = data
        
        return {
            'data': data_normalized.tolist(),
            'shape': data.shape,
            'rows_label': rows_label,
            'cols_label': cols_label,
            'original_range': (float(np.min(data)), float(np.max(data))),
            'normalized': normalize
        }
    
    @staticmethod
    def connectivity_matrix_data(
        weights: np.ndarray,
        neuron_ids: List[int] = None,
        threshold: float = 0.0
    ) -> Dict:
        """
        Prepare connectivity matrix data
        
        Args:
            weights: 2D array of connection weights (from x to)
            neuron_ids: Labels for neurons
            threshold: Threshold for including connections
            
        Returns:
            Dictionary with connectivity data
        """
        if neuron_ids is None:
            neuron_ids = list(range(weights.shape[0]))
        
        # Apply threshold
        masked_weights = weights.copy()
        masked_weights[np.abs(masked_weights) < threshold] = 0
        
        return {
            'connectivity_matrix': masked_weights.tolist(),
            'neuron_ids': neuron_ids,
            'n_neurons': len(neuron_ids),
            'connection_count': int(np.sum(masked_weights != 0)),
            'threshold': threshold,
            'weight_range': (float(np.min(weights)), float(np.max(weights)))
        }
    
    @staticmethod
    def psth_data(
        spike_times: np.ndarray,
        trial_starts: np.ndarray,
        bin_size: float = 0.01,
        pre_window: float = 0.5,
        post_window: float = 1.0
    ) -> Dict:
        """
        Prepare peristimulus time histogram (PSTH) data
        
        Args:
            spike_times: All spike times for a neuron
            trial_starts: Start times of trials (typically stimulus onset)
            bin_size: Histogram bin size in seconds
            pre_window: Time before stimulus onset to include (seconds)
            post_window: Time after stimulus onset to include (seconds)
            
        Returns:
            Dictionary with PSTH data
        """
        psth_matrix = []
        
        for trial_start in trial_starts:
            trial_window_start = trial_start - pre_window
            trial_window_end = trial_start + post_window
            
            # Get spikes in this trial window
            trial_spikes = spike_times[
                (spike_times >= trial_window_start) & 
                (spike_times <= trial_window_end)
            ]
            
            # Align to trial start
            trial_spikes_aligned = trial_spikes - trial_start
            psth_matrix.append(trial_spikes_aligned.tolist())
        
        # Compute histogram
        time_bins = np.arange(-pre_window, post_window + bin_size, bin_size)
        psth_hist, _ = np.histogram(
            np.concatenate(psth_matrix) if psth_matrix else [],
            bins=time_bins
        )
        
        return {
            'trial_aligned_spikes': psth_matrix,
            'histogram': psth_hist.tolist(),
            'bin_centers': (time_bins[:-1] + time_bins[1:]) / 2,
            'bin_size': bin_size,
            'n_trials': len(trial_starts),
            'pre_window': pre_window,
            'post_window': post_window
        }
    
    @staticmethod
    def tuning_curve_data(
        stimulus_values: np.ndarray,
        response_values: np.ndarray,
        error_bars: np.ndarray = None
    ) -> Dict:
        """
        Prepare tuning curve data
        
        Args:
            stimulus_values: Stimulus parameter values (e.g., orientations)
            response_values: Neural response values (e.g., firing rates)
            error_bars: Error estimates (std, sem, etc.)
            
        Returns:
            Dictionary with tuning curve data
        """
        return {
            'stimulus': stimulus_values.tolist() if isinstance(stimulus_values, np.ndarray) else stimulus_values,
            'response': response_values.tolist() if isinstance(response_values, np.ndarray) else response_values,
            'error_bars': error_bars.tolist() if error_bars is not None else None,
            'peak_response': float(np.max(response_values)),
            'peak_stimulus': float(stimulus_values[np.argmax(response_values)])
        }
    
    @staticmethod
    def neural_trajectory_data(
        embedding: np.ndarray,
        time_points: np.ndarray = None,
        trial_indices: np.ndarray = None,
        labels: Dict = None
    ) -> Dict:
        """
        Prepare neural trajectory data (e.g., from dimensionality reduction)
        
        Args:
            embedding: 2D or 3D projection of neural activity (n_timepoints x n_dims)
            time_points: Time indices for each point
            trial_indices: Trial indices for coloring
            labels: Dictionary mapping indices to labels
            
        Returns:
            Dictionary with trajectory data
        """
        n_dims = embedding.shape[1]
        
        if time_points is None:
            time_points = np.arange(embedding.shape[0])
        
        return {
            'embedding': embedding.tolist(),
            'n_dimensions': n_dims,
            'n_timepoints': embedding.shape[0],
            'time_points': time_points.tolist() if isinstance(time_points, np.ndarray) else time_points,
            'trial_indices': trial_indices.tolist() if trial_indices is not None else None,
            'labels': labels
        }


class AnalysisVisualizer:
    """Generate visualization specifications for analysis results"""
    
    @staticmethod
    def create_analysis_figure(
        figure_type: str,
        data: Dict,
        title: str = "",
        figure_size: Tuple[int, int] = (12, 8)
    ) -> Dict:
        """
        Create a complete figure specification
        
        Args:
            figure_type: Type of figure (raster, heatmap, tuning_curve, etc.)
            data: Data prepared by NeuroscienceVisualizations
            title: Figure title
            figure_size: Figure size in inches
            
        Returns:
            Dictionary with complete figure specification
        """
        figure_spec = {
            'type': figure_type,
            'title': title,
            'size': figure_size,
            'data': data,
            'created_at': None
        }
        
        # Add type-specific defaults
        if figure_type == 'raster':
            figure_spec.update({
                'xlabel': 'Time (s)',
                'ylabel': 'Neuron Index',
                'marker_size': 2
            })
        elif figure_type == 'heatmap':
            figure_spec.update({
                'colormap': 'viridis',
                'xlabel': data.get('cols_label', 'Time'),
                'ylabel': data.get('rows_label', 'Neurons'),
                'colorbar_label': 'Activity'
            })
        elif figure_type == 'tuning_curve':
            figure_spec.update({
                'xlabel': 'Stimulus Parameter',
                'ylabel': 'Response (Hz)',
                'line_width': 2
            })
        
        return figure_spec


if __name__ == "__main__":
    # Example usage
    viz = NeuroscienceVisualizations()
    
    # Example: Create spike raster data
    print("Creating spike raster data...")
    spike_times_1 = np.array([0.1, 0.3, 0.5, 0.8])
    spike_times_2 = np.array([0.15, 0.4, 0.9])
    raster = viz.spike_raster_data([spike_times_1, spike_times_2])
    print(f"Raster with {len(raster['neurons'])} neurons")
    
    # Example: Create heatmap data
    print("\nCreating heatmap data...")
    neural_response = np.random.randn(50, 100)
    heatmap = viz.heatmap_data(neural_response, rows_label="Neurons", cols_label="Time")
    print(f"Heatmap shape: {heatmap['shape']}")
    
    # Example: Create tuning curve data
    print("\nCreating tuning curve data...")
    orientations = np.arange(0, 180, 15)
    responses = np.array([10, 25, 45, 60, 50, 35, 20, 15, 18, 30, 55, 70])
    tuning = viz.tuning_curve_data(orientations, responses)
    print(f"Peak response at {tuning['peak_stimulus']:.0f}°")
