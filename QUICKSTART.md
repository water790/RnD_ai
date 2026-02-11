# Quick Start Guide for Neuroscience R&D Assistant

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key
```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Or create .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Test Installation
```bash
python -c "from main import NeuroscienceRnDAssistant; print('Ready!')"
```

## Common Tasks

### Create an Experiment
```python
from main import NeuroscienceRnDAssistant

assistant = NeuroscienceRnDAssistant()
exp = assistant.create_experiment(
    experiment_id="EXP_001",
    title="My experiment",
    organism="Mus musculus",
    brain_region="Visual Cortex",
    technique="Electrophysiology",
    pi="Dr. Smith"
)
print(f"Created: {exp.title}")
```

### Analyze Spike Data
```python
from main import AnalysisTools
import numpy as np

spike_times = np.array([0.05, 0.12, 0.15, 0.23, 0.31])
firing_rates, bins = AnalysisTools.compute_firing_rate(spike_times)
print(f"Mean firing rate: {firing_rates.mean():.1f} Hz")
```

### Get LLM Analysis
```python
from llm_integration import GPTAdapter, NeuroscienceRnDClient

llm = NeuroscienceRnDClient(GPTAdapter())
result = llm.analyze_data(
    experiment_context="V1 recording, orientation tuning",
    data_summary="Peak response at 90 degrees, 45 Hz",
    question="What does this tell us about visual processing?"
)
print(result)
```

### Design an Experiment
```python
llm = NeuroscienceRnDClient(GPTAdapter())
design = llm.design_experiment(
    background="Understanding motion processing in MT",
    objective="What's the best way to measure direction selectivity?"
)
print(design)
```

### Generate Hypotheses
```python
hypotheses = llm.generate_hypotheses(
    background="V1 neurons are direction selective",
    observation="Direction selectivity decreases with stimulus speed",
    focus="Why does this happen?"
)
print(hypotheses)
```

### Prepare Data for Visualization
```python
from visualization import NeuroscienceVisualizations
import numpy as np

viz = NeuroscienceVisualizations()

# Spike raster
spike_times = [np.random.uniform(0, 10, 20) for _ in range(10)]
raster = viz.spike_raster_data(spike_times)

# Tuning curve
orientations = np.linspace(0, 180, 13)
responses = 50 * np.sin((orientations - 90) * np.pi / 180) + 50
tuning = viz.tuning_curve_data(orientations, responses)

# Heatmap
neural_data = np.random.randn(30, 100)
heatmap = viz.heatmap_data(neural_data)
```

## Running Examples

```bash
# Run all example workflows
python workflows.py

# Import specific workflow
python -c "from workflows import workflow_hypothesis_generation; workflow_hypothesis_generation()"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named 'openai'" | Run: `pip install openai` |
| "No API key found" | Set OPENAI_API_KEY environment variable |
| "API rate limit exceeded" | Wait before retrying; use gpt-3.5-turbo instead |
| "Memory error with large data" | Process data in chunks; use HDF5 format |

## Next Steps

1. **Customize Configuration**: Copy `config_template.py` to `config.py` and edit settings
2. **Load Your Data**: Use `DataHandler.load_timeseries()` to import data
3. **Run Analysis**: Use `AnalysisTools` for computational analysis
4. **Get LLM Help**: Use `NeuroscienceRnDClient` for interpretation
5. **Save Results**: Export with `assistant.save_experiment_metadata()`

## File Organization

```
project/
├── data/           # Your experimental data
├── outputs/        # Analysis results and exports
├── logs/           # Log files
├── main.py         # Core functionality
├── llm_integration.py  # LLM interface
├── visualization.py    # Visualization utilities
└── workflows.py        # Example workflows
```

## Example Data Format

### CSV (timeseries)
```
0.001,25.3,15.2,8.9
0.002,25.1,15.1,8.8
0.003,25.4,15.3,9.1
```

### NPZ (with metadata)
```python
import numpy as np
data = np.random.randn(100, 1000)
metadata = {'sampling_rate': 30000, 'units': 'μV'}
np.savez('recording.npz', data=data, **metadata)
```

## Performance Tips

- **Small datasets** (<100MB): Use CSV format
- **Medium datasets** (100MB-1GB): Use NPZ format
- **Large datasets** (>1GB): Use HDF5 format
- **Real-time analysis**: Use streaming with chunked processing
- **LLM calls**: Cache responses to minimize API calls

## Security

⚠️ **Important**: Never commit API keys to version control!

```bash
# Add to .gitignore
.env
.env.local
*.key
config.py
```

## Support Resources

- **Documentation**: See README.md
- **Examples**: Run workflows.py
- **API Docs**: Check docstrings in code
- **OpenAI API**: https://platform.openai.com/docs

## First Research Workflow

```python
# 1. Setup
from main import NeuroscienceRnDAssistant
from llm_integration import NeuroscienceRnDClient, GPTAdapter

assistant = NeuroscienceRnDAssistant()
llm = NeuroscienceRnDClient(GPTAdapter())

# 2. Design experiment
design = llm.design_experiment(
    background="Study spatial attention in V1",
    objective="How does attention modulate visual responses?"
)

# 3. Create experiment record
exp = assistant.create_experiment(
    experiment_id="EXP_001",
    title="Attention in V1",
    organism="Mus musculus",
    brain_region="Primary Visual Cortex",
    technique="Two-photon imaging",
    pi="Your Name"
)

# 4. Analyze your data
interpretation = llm.analyze_data(
    experiment_context=f"{exp.title} in {exp.brain_region}",
    data_summary="Attended stimuli evoked 40% stronger responses",
    question="What mechanisms explain attention modulation?"
)

# 5. Save results
assistant.save_experiment_metadata("outputs/exp_001.json")
```

---
For more details, see the full [README.md](README.md)
