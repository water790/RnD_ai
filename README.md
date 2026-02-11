# Neuroscience R&D Assistant for GPT

A comprehensive Python framework for neuroscience research and development with integrated LLM (Large Language Model) support for GPT variants. This toolkit provides researchers with AI-assisted tools for experimental design, data analysis, hypothesis generation, and literature review.

## Features

### Core Capabilities

- **Experiment Management**: Track and organize neuroscience experiments with metadata
- **Data Handling**: Support for multiple neuroscience data formats (CSV, NPZ, HDF5)
- **Analysis Tools**:
  - Firing rate computation
  - Cross-correlation analysis
  - Population-level metrics from spike rasters
  - Raster metrics computation

- **LLM Integration**: Seamless integration with OpenAI GPT models for:
  - Experimental design assistance
  - Data analysis and interpretation
  - Hypothesis generation
  - Literature review synthesis
  - Methodology review and critique

- **Knowledge Management**: Build and query domain-specific knowledge bases
- **Visualization**: Prepare data for neuroscience-specific visualizations
- **Statistical Support**: Sample size recommendations and power analysis

### Research Tasks Supported

1. **Data Analysis** - Analyze neural recordings with LLM-assisted interpretation
2. **Experimental Design** - Get AI-powered suggestions for experiment planning
3. **Literature Review** - Synthesize research topics with LLM assistance
4. **Hypothesis Generation** - Generate testable hypotheses from observations
5. **Result Interpretation** - Understand findings in context of neuroscience theory
6. **Methodology Review** - Get feedback on experimental approaches

## Project Structure

```
RnD_ai/
├── main.py                 # Core assistant and analysis tools
├── llm_integration.py      # LLM adapter and prompt building
├── visualization.py        # Neuroscience visualization utilities
├── workflows.py            # Example research workflows
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for LLM features)

### Setup

1. Clone or download this project:
```bash
cd RnD_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Create .env file in project directory
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Quick Start

### Basic Usage

```python
from main import NeuroscienceRnDAssistant, AnalysisTools
from llm_integration import GPTAdapter, NeuroscienceRnDClient
import numpy as np

# Initialize assistant and LLM client
assistant = NeuroscienceRnDAssistant()
gpt_adapter = GPTAdapter(model="gpt-4")
llm_client = NeuroscienceRnDClient(gpt_adapter)

# Create an experiment
exp = assistant.create_experiment(
    experiment_id="EXP_001",
    title="Single-unit recording in V1",
    organism="Mus musculus",
    brain_region="Visual Cortex (V1)",
    technique="Electrophysiology",
    description="Recording from layer 2/3 neurons",
    pi="Dr. Smith"
)

# Perform analysis
spike_times = np.array([0.05, 0.12, 0.15, 0.23])
firing_rates, time_bins = AnalysisTools.compute_firing_rate(spike_times)

# Get LLM assistance with data interpretation
interpretation = llm_client.interpret_results(
    experiment_context="V1 single-unit recording",
    results=f"Mean firing rate: {firing_rates.mean():.1f} Hz"
)
print(interpretation)
```

### Running Example Workflows

```bash
python workflows.py
```

This will run several example workflows demonstrating:
- Experiment design with LLM assistance
- Data analysis and interpretation
- Hypothesis generation
- Knowledge base management
- Visualization preparation

## Module Documentation

### main.py

Core module containing:

**NeuroscienceRnDAssistant**
- `create_experiment()` - Create new experiment record
- `save_experiment_metadata()` - Save experiments to JSON

**AnalysisTools**
- `compute_firing_rate()` - Compute firing rates from spike times
- `compute_cross_correlation()` - Cross-correlation between signals
- `compute_raster_metrics()` - Population-level metrics

**DataHandler**
- `load_timeseries()` - Load electrophysiology/imaging data
- `save_timeseries()` - Save data with metadata

**ExperimentDesigner**
- `suggest_sample_size()` - Statistical power analysis
- `create_experimental_plan()` - Template for experiment planning

**KnowledgeBase**
- `add_entry()` - Add knowledge base entries
- `search_by_tag()` - Find entries by tag
- `export_knowledge_base()` - Export to JSON

### llm_integration.py

LLM integration module:

**GPTAdapter**
- `generate_response()` - Get text response from GPT
- `generate_structured()` - Get structured JSON response

**NeurosciencePromptBuilder**
- `build_analysis_prompt()` - Create analysis prompts
- `build_literature_prompt()` - Create literature review prompts
- `build_methodology_review_prompt()` - Create methodology review prompts

**NeuroscienceRnDClient**
- `analyze_data()` - Data analysis with LLM
- `design_experiment()` - Experimental design assistance
- `interpret_results()` - Results interpretation
- `generate_hypotheses()` - Hypothesis generation
- `review_literature()` - Literature review assistance
- `review_methodology()` - Methodology review

### visualization.py

Visualization utilities:

**NeuroscienceVisualizations**
- `spike_raster_data()` - Prepare raster plot data
- `heatmap_data()` - Prepare heatmap data
- `connectivity_matrix_data()` - Prepare connectivity data
- `psth_data()` - Prepare PSTH data
- `tuning_curve_data()` - Prepare tuning curve data
- `neural_trajectory_data()` - Prepare trajectory data

**AnalysisVisualizer**
- `create_analysis_figure()` - Create figure specifications

## Usage Examples

### Example 1: Experimental Design

```python
from llm_integration import NeuroscienceRnDClient, GPTAdapter

llm_client = NeuroscienceRnDClient(GPTAdapter())

design = llm_client.design_experiment(
    background="Understanding visual processing in V1",
    objective="Design a study to investigate direction selectivity"
)
print(design)
```

### Example 2: Data Analysis and Interpretation

```python
from main import AnalysisTools
from llm_integration import NeuroscienceRnDClient, GPTAdapter
import numpy as np

# Compute metrics
spike_matrix = np.random.binomial(1, 0.01, size=(20, 1000))
metrics = AnalysisTools.compute_raster_metrics(spike_matrix)

# Get LLM interpretation
llm_client = NeuroscienceRnDClient(GPTAdapter())
interpretation = llm_client.interpret_results(
    experiment_context="Population recording from motor cortex",
    results=f"Mean firing rate: {metrics['mean_firing_rate']:.2f} Hz"
)
print(interpretation)
```

### Example 3: Hypothesis Generation

```python
llm_client = NeuroscienceRnDClient(GPTAdapter())

hypotheses = llm_client.generate_hypotheses(
    background="Motor cortex neurons show increased synchrony during movement",
    observation="Synchrony is actually higher during motor preparation than execution",
    focus="What mechanisms might explain this?"
)
print(hypotheses)
```

### Example 4: Visualization Data Preparation

```python
from visualization import NeuroscienceVisualizations
import numpy as np

viz = NeuroscienceVisualizations()

# Prepare raster data
spike_times = [np.array([0.1, 0.3, 0.5]), np.array([0.15, 0.4])]
raster = viz.spike_raster_data(spike_times)

# Prepare tuning curve data
orientations = np.linspace(0, 180, 13)
responses = np.array([10, 25, 45, 60, 50, 35, 20, 15, 18, 30, 55, 70])
tuning = viz.tuning_curve_data(orientations, responses)
```

## Supported Neuroscience Techniques

- **Electrophysiology**: Single-unit and multi-unit recordings
- **Calcium Imaging**: Two-photon and widefield imaging
- **fMRI**: Functional magnetic resonance imaging
- **EEG/MEG**: Whole-brain electrophysiology
- **Optogenetics**: Combination with electrical/optical techniques

## Data Format Support

- **CSV**: Comma-separated timeseries data
- **NPZ**: NumPy compressed arrays with metadata
- **HDF5**: Large-scale data storage (preparation for future support)

## Statistical Methods

- Firing rate computation (multiple window sizes)
- Cross-correlation analysis
- Population synchrony measures
- Sample size estimation
- Power analysis

## LLM Models Supported

- GPT-4 (recommended for complex analysis)
- GPT-3.5-turbo (faster, lower cost)
- Custom endpoints compatible with OpenAI API

## Extensibility

The framework is designed to be extensible:

1. **Custom Adapters**: Inherit from `BaseLLMAdapter` to support other LLM providers
2. **Custom Analysis Tools**: Add new analysis methods to `AnalysisTools`
3. **Custom Visualizations**: Extend `NeuroscienceVisualizations`
4. **Custom Workflows**: Create new workflows in `workflows.py`

Example custom adapter:

```python
from llm_integration import BaseLLMAdapter

class CustomLLMAdapter(BaseLLMAdapter):
    def generate_response(self, prompt: str, **kwargs) -> str:
        # Your implementation here
        pass
    
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        # Your implementation here
        pass

# Use with client
custom_adapter = CustomLLMAdapter()
client = NeuroscienceRnDClient(custom_adapter)
```

## Best Practices

### When Using LLM for Analysis

1. **Provide Context**: Include relevant experimental details in prompts
2. **Verify Results**: Cross-reference LLM suggestions with literature
3. **Use Structured Prompts**: Use the built-in prompt builders for consistency
4. **Store Results**: Export conversations for reproducibility
5. **Version Control**: Track which model/version generated analyses

### Data Handling

1. **Preprocess Data**: Clean and validate before analysis
2. **Document Formats**: Clearly specify data format and units
3. **Preserve Raw Data**: Keep original files for re-analysis
4. **Metadata**: Store experimental parameters with data

## Performance Considerations

- Large datasets (>10GB) should be streamed using HDF5
- Firing rate computation scales linearly with data size
- Cross-correlation is O(n log n) with FFT implementation
- LLM calls are rate-limited; batch requests when possible

## Troubleshooting

### Issue: "No API key found"
**Solution**: Set OPENAI_API_KEY environment variable or create .env file

### Issue: API rate limits exceeded
**Solution**: Implement exponential backoff; use gpt-3.5-turbo for less critical tasks

### Issue: Memory error with large spike matrices
**Solution**: Process data in chunks; use sparse matrices for sparse data

## Citation

If you use this toolkit in research, please cite:

```bibtex
@software{neurosci_rnd_gpt,
  title={Neuroscience R&D Assistant for GPT},
  author={Your Name},
  year={2024},
  url={https://github.com/yourname/RnD_ai}
}
```

## Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Support for additional LLM providers (Claude, open-source models)
- [ ] Advanced statistical methods (Bayesian analysis, machine learning)
- [ ] Real-time data streaming
- [ ] Interactive visualization integration
- [ ] Database backend for experiment tracking
- [ ] Publication-ready figure generation

## License

MIT License - See LICENSE file for details

## References

### Neuroscience Methods
- Spike detection and sorting
- Receptive field mapping
- Population coding theory
- Neural circuits and dynamics

### LLM Integration
- OpenAI API documentation
- Prompt engineering best practices
- Structured output (JSON) generation

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the examples in `workflows.py`
- Review the inline documentation in each module

## Future Roadmap

- [ ] Web UI for experiment management
- [ ] Real-time collaboration features
- [ ] Integration with data repositories (Allen Institute, OpenNeuro)
- [ ] Advanced machine learning for pattern discovery
- [ ] Publication pipeline integration
- [ ] Multi-modal LLM support (analyze images, videos)

---

**Last Updated**: February 2026
**Version**: 1.0.0
