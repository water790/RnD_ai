# API Reference - Neuroscience R&D Assistant

## Table of Contents
1. [NeuroscienceRnDAssistant](#neuroscience-rnd-assistant)
2. [AnalysisTools](#analysis-tools)
3. [DataHandler](#data-handler)
4. [ExperimentDesigner](#experiment-designer)
5. [KnowledgeBase](#knowledge-base)
6. [LLM Integration](#llm-integration)
7. [Visualization](#visualization)

---

## NeuroscienceRnDAssistant

Main class for managing neuroscience research workflows.

### Constructor
```python
NeuroscienceRnDAssistant(api_key: str = None, model: str = "gpt-4")
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable.
- `model` (str): Model to use. Default: "gpt-4"

**Example:**
```python
assistant = NeuroscienceRnDAssistant()
assistant = NeuroscienceRnDAssistant(api_key="sk-...", model="gpt-3.5-turbo")
```

### Methods

#### create_experiment()
```python
def create_experiment(
    experiment_id: str,
    title: str,
    organism: str,
    brain_region: str,
    technique: str,
    description: str = "",
    pi: str = "",
    protocol_url: str = None,
    notes: str = None
) -> ExperimentMetadata
```

Create a new experiment record.

**Parameters:**
- `experiment_id` (str): Unique identifier for the experiment
- `title` (str): Descriptive title
- `organism` (str): Species (e.g., "Mus musculus")
- `brain_region` (str): Target brain region
- `technique` (str): Recording/imaging technique
- `description` (str): Detailed description
- `pi` (str): Principal investigator name
- `protocol_url` (str): URL to experimental protocol
- `notes` (str): Additional notes

**Returns:** ExperimentMetadata object

**Example:**
```python
exp = assistant.create_experiment(
    experiment_id="EXP_001",
    title="V1 orientation tuning",
    organism="Mus musculus",
    brain_region="Visual Cortex (V1)",
    technique="Two-photon imaging",
    description="Characterizing orientation selectivity in layer 2/3",
    pi="Dr. Smith"
)
```

#### save_experiment_metadata()
```python
def save_experiment_metadata(output_path: str) -> None
```

Save all experiments to JSON file.

**Parameters:**
- `output_path` (str): Path to output JSON file

**Example:**
```python
assistant.save_experiment_metadata("outputs/experiments.json")
```

---

## AnalysisTools

Collection of neuroscience analysis methods.

### compute_firing_rate()
```python
@staticmethod
def compute_firing_rate(
    spike_times: np.ndarray,
    window_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]
```

Compute firing rate from spike times using sliding window.

**Parameters:**
- `spike_times` (np.ndarray): Array of spike times in seconds
- `window_size` (float): Window size in seconds. Default: 0.1

**Returns:** Tuple of (firing_rates, time_bins)

**Example:**
```python
spike_times = np.array([0.05, 0.12, 0.15, 0.23])
firing_rates, bins = AnalysisTools.compute_firing_rate(spike_times, window_size=0.05)
print(f"Mean FR: {firing_rates.mean():.1f} Hz")
```

### compute_cross_correlation()
```python
@staticmethod
def compute_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag: int = 100
) -> np.ndarray
```

Compute cross-correlation between two signals.

**Parameters:**
- `signal1` (np.ndarray): First signal
- `signal2` (np.ndarray): Second signal
- `max_lag` (int): Maximum lag in samples. Default: 100

**Returns:** Cross-correlation coefficients

**Example:**
```python
signal1 = np.random.randn(1000)
signal2 = np.random.randn(1000)
xcorr = AnalysisTools.compute_cross_correlation(signal1, signal2)
```

### compute_raster_metrics()
```python
@staticmethod
def compute_raster_metrics(spike_matrix: np.ndarray) -> Dict[str, Any]
```

Compute population-level metrics from spike raster.

**Parameters:**
- `spike_matrix` (np.ndarray): Shape (n_neurons, n_timepoints), 1=spike, 0=no spike

**Returns:** Dictionary with metrics:
- `total_spikes` (int): Total number of spikes
- `mean_firing_rate` (float): Population mean firing rate (Hz)
- `spikes_per_neuron` (list): Number of spikes per neuron
- `population_synchrony` (float): Mean correlation between neurons
- `n_neurons` (int): Number of neurons
- `n_timepoints` (int): Number of timepoints

**Example:**
```python
spike_matrix = np.random.binomial(1, 0.01, size=(50, 1000))
metrics = AnalysisTools.compute_raster_metrics(spike_matrix)
print(f"Mean FR: {metrics['mean_firing_rate']:.2f} Hz")
```

---

## DataHandler

Data loading and saving utilities.

### load_timeseries()
```python
@staticmethod
def load_timeseries(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]
```

Load timeseries data from file.

**Parameters:**
- `filepath` (str): Path to data file (.csv, .npz, .h5)

**Returns:** Tuple of (data_array, metadata_dict)

**Supported Formats:**
- CSV: Tab or comma-separated values
- NPZ: NumPy compressed format with metadata
- HDF5: Large-scale data (future support)

**Example:**
```python
data, metadata = DataHandler.load_timeseries("data/recording.npz")
print(f"Shape: {data.shape}")
print(f"Sampling rate: {metadata.get('sampling_rate', 'unknown')}")
```

### save_timeseries()
```python
@staticmethod
def save_timeseries(
    data: np.ndarray,
    filepath: str,
    metadata: Dict = None
) -> None
```

Save timeseries data with metadata.

**Parameters:**
- `data` (np.ndarray): Data array to save
- `filepath` (str): Output path (.csv or .npz)
- `metadata` (Dict): Optional metadata dictionary

**Example:**
```python
data = np.random.randn(100, 1000)
metadata = {'sampling_rate': 30000, 'units': 'μV'}
DataHandler.save_timeseries(data, "outputs/processed.npz", metadata)
```

---

## ExperimentDesigner

Experimental design and planning utilities.

### suggest_sample_size()
```python
@staticmethod
def suggest_sample_size(
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    test_type: str = "t-test"
) -> Dict[str, Any]
```

Suggest sample size for statistical test.

**Parameters:**
- `effect_size` (float): Expected effect size (Cohen's d). Default: 0.5
- `alpha` (float): Significance level. Default: 0.05
- `power` (float): Statistical power (1 - β). Default: 0.8
- `test_type` (str): Type of test. Default: "t-test"

**Returns:** Dictionary with recommendations:
- `suggested_n_per_group` (int): Sample size per group
- `total_n` (int): Total sample size
- `effect_size`, `alpha`, `power` (floats)

**Example:**
```python
rec = ExperimentDesigner.suggest_sample_size(effect_size=0.8, power=0.9)
print(f"Need {rec['total_n']} animals total")
```

### create_experimental_plan()
```python
@staticmethod
def create_experimental_plan(
    hypothesis: str,
    organism: str,
    brain_region: str,
    expected_duration_hours: float,
    n_animals: int
) -> Dict[str, Any]
```

Create experimental plan template.

**Parameters:**
- `hypothesis` (str): Research hypothesis
- `organism` (str): Species name
- `brain_region` (str): Target region
- `expected_duration_hours` (float): Duration per animal (hours)
- `n_animals` (int): Number of animals

**Returns:** Plan dictionary with timeline, data requirements, analysis steps

**Example:**
```python
plan = ExperimentDesigner.create_experimental_plan(
    hypothesis="Layer 2/3 maintains direction selectivity",
    organism="Mus musculus",
    brain_region="V1",
    expected_duration_hours=4.0,
    n_animals=5
)
```

---

## KnowledgeBase

Domain knowledge management.

### Constructor
```python
KnowledgeBase()
```

### add_entry()
```python
def add_entry(
    key: str,
    content: str,
    tags: List[str] = None,
    metadata: Dict = None
) -> None
```

Add entry to knowledge base.

**Parameters:**
- `key` (str): Unique identifier
- `content` (str): Entry content/text
- `tags` (List[str]): Optional tags for organization
- `metadata` (Dict): Optional metadata

**Example:**
```python
kb = KnowledgeBase()
kb.add_entry(
    key="technique_two_photon",
    content="Two-photon microscopy allows...",
    tags=["imaging", "techniques"],
    metadata={"type": "technique"}
)
```

### search_by_tag()
```python
def search_by_tag(tag: str) -> List[str]
```

Find all entries with a given tag.

**Parameters:**
- `tag` (str): Tag to search for

**Returns:** List of matching entry keys

**Example:**
```python
imaging_entries = kb.search_by_tag("imaging")
```

### get_entry()
```python
def get_entry(key: str) -> Optional[Dict]
```

Retrieve an entry from the knowledge base.

**Parameters:**
- `key` (str): Entry key

**Returns:** Entry dictionary or None if not found

---

## LLM Integration

### GPTAdapter

OpenAI GPT adapter.

#### Constructor
```python
GPTAdapter(api_key: str = None, model: str = "gpt-4")
```

#### generate_response()
```python
def generate_response(
    prompt: str,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs
) -> str
```

Generate text response from GPT.

**Parameters:**
- `prompt` (str): Input prompt
- `temperature` (float): Sampling temperature (0.0-1.0)
- `max_tokens` (int): Maximum response length
- `**kwargs`: Additional OpenAI API parameters

**Returns:** Generated text response

**Example:**
```python
gpt = GPTAdapter(model="gpt-4")
response = gpt.generate_response("Explain orientation selectivity")
```

### NeuroscienceRnDClient

High-level client for research tasks.

#### Constructor
```python
NeuroscienceRnDClient(llm_adapter: BaseLLMAdapter)
```

#### analyze_data()
```python
def analyze_data(
    experiment_context: str,
    data_summary: str,
    question: str
) -> str
```

Analyze experimental data with LLM assistance.

**Example:**
```python
client = NeuroscienceRnDClient(GPTAdapter())
analysis = client.analyze_data(
    experiment_context="V1 recordings during visual stimulus",
    data_summary="Peak response at 90° orientation, 45 Hz",
    question="What does this tell us about V1 processing?"
)
```

#### design_experiment()
```python
def design_experiment(
    background: str,
    objective: str
) -> str
```

Get experimental design assistance from LLM.

#### interpret_results()
```python
def interpret_results(
    experiment_context: str,
    results: str,
    specific_question: str = ""
) -> str
```

Interpret experimental results with LLM.

#### generate_hypotheses()
```python
def generate_hypotheses(
    background: str,
    observation: str,
    focus: str = ""
) -> str
```

Generate testable hypotheses from observations.

#### review_literature()
```python
def review_literature(
    topic: str,
    keywords: List[str],
    focus: str = ""
) -> str
```

Get literature review synthesis from LLM.

#### review_methodology()
```python
def review_methodology(
    methodology: str,
    setup: str,
    concerns: List[str] = None
) -> str
```

Review experimental methodology and get suggestions.

---

## Visualization

### NeuroscienceVisualizations

Neuroscience visualization data preparation.

#### spike_raster_data()
```python
@staticmethod
def spike_raster_data(
    spike_times_list: List[np.ndarray],
    trial_starts: np.ndarray = None,
    stimulus_window: Tuple[float, float] = None
) -> Dict
```

Prepare spike raster plot data.

**Parameters:**
- `spike_times_list` (List[np.ndarray]): Spike times for each neuron
- `trial_starts` (np.ndarray): Trial start times
- `stimulus_window` (Tuple): (start, end) times relative to trial

**Returns:** Dictionary with raster data

**Example:**
```python
viz = NeuroscienceVisualizations()
spike_times = [np.array([0.1, 0.3]), np.array([0.15, 0.4])]
raster = viz.spike_raster_data(spike_times)
```

#### heatmap_data()
```python
@staticmethod
def heatmap_data(
    data: np.ndarray,
    rows_label: str = "Neurons",
    cols_label: str = "Time",
    normalize: bool = True
) -> Dict
```

Prepare heatmap visualization data (e.g., neural responses).

#### tuning_curve_data()
```python
@staticmethod
def tuning_curve_data(
    stimulus_values: np.ndarray,
    response_values: np.ndarray,
    error_bars: np.ndarray = None
) -> Dict
```

Prepare tuning curve data.

#### psth_data()
```python
@staticmethod
def psth_data(
    spike_times: np.ndarray,
    trial_starts: np.ndarray,
    bin_size: float = 0.01,
    pre_window: float = 0.5,
    post_window: float = 1.0
) -> Dict
```

Prepare peristimulus time histogram (PSTH) data.

#### connectivity_matrix_data()
```python
@staticmethod
def connectivity_matrix_data(
    weights: np.ndarray,
    neuron_ids: List[int] = None,
    threshold: float = 0.0
) -> Dict
```

Prepare connectivity matrix data.

#### neural_trajectory_data()
```python
@staticmethod
def neural_trajectory_data(
    embedding: np.ndarray,
    time_points: np.ndarray = None,
    trial_indices: np.ndarray = None,
    labels: Dict = None
) -> Dict
```

Prepare neural trajectory data (e.g., from dimensionality reduction).

---

## Data Types

### ExperimentMetadata
```python
@dataclass
class ExperimentMetadata:
    experiment_id: str
    title: str
    description: str
    organism: str
    brain_region: str
    technique: str
    timestamp: str
    principal_investigator: str
    protocol_url: Optional[str] = None
    notes: Optional[str] = None
```

### ResearchTask Enum
```python
class ResearchTask(Enum):
    DATA_ANALYSIS = "data_analysis"
    EXPERIMENTAL_DESIGN = "experimental_design"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    RESULT_INTERPRETATION = "result_interpretation"
    METHODOLOGY_REVIEW = "methodology_review"
    PUBLICATION_ASSISTANCE = "publication_assistance"
```

---

## Error Handling

All methods handle errors gracefully:
- Missing files logged and return None or empty dicts
- API errors logged with retry logic
- Invalid parameters raise TypeError or ValueError
- Missing API keys result in informative warnings

---

## Rate Limiting

LLM calls are rate-limited:
- Default: 10 calls per 60 seconds
- Exponential backoff on rate limit errors
- Response caching to reduce redundant calls

---

## Thread Safety

**Not thread-safe.** For concurrent use:
- Create separate instances per thread
- Use thread locks for shared assistant instance
- Consider async version for future releases

---

**API Reference Version**: 1.0
**Last Updated**: February 2026
