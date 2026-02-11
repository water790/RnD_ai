# Neuroscience R&D Assistant - Architecture & Design Summary

## Project Overview

A comprehensive Python framework for neuroscience research and development with integrated GPT-4 support. This assistant helps neuroscientists with experimental design, data analysis, hypothesis generation, and research interpretation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface Layer                          │
│                  (Jupyter, Scripts, CLI)                         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Integration Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  GPTAdapter  │  │ PromptBuilder│  │ NeuroscienceRnDClient│  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Core Analysis Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │AnalysisTools│  │DataHandler   │  │ExperimentDesigner    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              Support & Infrastructure Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ KnowledgeBase│  │Visualization │  │ExperimentMetadata    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### 1. **main.py** - Core Analysis Engine
   - **NeuroscienceRnDAssistant**: Main orchestrator for research workflows
   - **AnalysisTools**: Neuroscience-specific analysis (firing rates, correlation, population metrics)
   - **DataHandler**: Data loading and saving in multiple formats
   - **ExperimentDesigner**: Experiment planning and statistical recommendations
   - **KnowledgeBase**: Domain knowledge storage and retrieval
   - **ExperimentMetadata**: Structured experiment information

### 2. **llm_integration.py** - LLM Communication Layer
   - **BaseLLMAdapter**: Abstract interface for LLM providers
   - **GPTAdapter**: OpenAI GPT implementation
   - **NeurosciencePromptBuilder**: Specialized prompt templates
   - **NeuroscienceRnDClient**: High-level client for research tasks
   - **ResearchTask**: Enum of supported research activities

### 3. **visualization.py** - Data Visualization Support
   - **NeuroscienceVisualizations**: Prepare neuroscience-specific visualizations
   - **AnalysisVisualizer**: Generate figure specifications
   - Supports: rasters, heatmaps, connectivity, PSTH, tuning curves, trajectories

### 4. **workflows.py** - Example Research Workflows
   - Demonstrates end-to-end research processes
   - Combines analysis, LLM interpretation, and knowledge management
   - Serves as templates for users' own workflows

## Data Flow

### Experiment Creation Flow
```
User Input → NeuroscienceRnDAssistant.create_experiment()
           → ExperimentMetadata (dataclass)
           → Stored in experiments dict
           → Can be exported to JSON
```

### Data Analysis Flow
```
Raw Data (CSV/NPZ) → DataHandler.load_timeseries()
                   → NumPy array + metadata
                   → AnalysisTools (computation)
                   → Results dictionary
                   → Export or visualization prep
```

### LLM-Assisted Analysis Flow
```
Experiment Context + Data Summary
    ↓
NeurosciencePromptBuilder.build_X_prompt()
    ↓
GPTAdapter.generate_response()
    ↓
OpenAI API (GPT-4/3.5-turbo)
    ↓
Response Text/JSON
    ↓
NeuroscienceRnDClient (storage + export)
```

## Key Design Patterns

### 1. **Adapter Pattern**
- `BaseLLMAdapter` allows swapping different LLM providers
- Currently supports OpenAI, extensible to other providers

### 2. **Strategy Pattern**
- `ResearchTask` enum determines which prompt/analysis strategy to use
- Different specialized prompts for different research phases

### 3. **Builder Pattern**
- `NeurosciencePromptBuilder` constructs complex, domain-specific prompts
- `ExperimentDesigner` builds experimental plans

### 4. **Dataclass Pattern**
- `ExperimentMetadata` for structured experiment information
- Easily serializable to JSON for reproducibility

### 5. **Facade Pattern**
- `NeuroscienceRnDClient` provides simplified interface to complex LLM operations
- Abstracts prompt building and response handling

## Supported Neuroscience Domains

### Recording Techniques
- Electrophysiology (single-unit, multi-unit, MEA)
- Calcium imaging (two-photon, widefield)
- fMRI/MEG/EEG
- Optogenetics
- Patch-clamp

### Analysis Methods
- Firing rate computation
- Cross-correlation
- Population synchrony
- Raster analysis
- Connectivity matrices
- Dimensionality reduction (trajectory data)

### Research Tasks
1. Experimental Design
2. Data Analysis & Interpretation
3. Hypothesis Generation
4. Literature Review
5. Methodology Review
6. Publication Assistance

## Data Format Support

| Format | Use Case | Size Limit |
|--------|----------|-----------|
| CSV | Simple timeseries | <100MB |
| NPZ | Compressed with metadata | 100MB-1GB |
| HDF5 | Large-scale data | >1GB |

## Configuration System

### Three-Level Configuration
1. **Environment Variables** (.env file)
2. **Configuration Module** (config.py)
3. **Runtime Parameters** (function arguments)

Priority: Runtime > Config > Environment

### Configuration Areas
- LLM settings (model, temperature, tokens)
- Data paths (input, output, cache)
- Analysis parameters (window sizes, thresholds)
- Experiment defaults (organism, region, technique)
- Logging and caching

## Integration Points

### External Services
- **OpenAI API**: GPT-4/3.5-turbo for analysis and interpretation
- Potential: Claude, open-source LLMs, local models

### Data Sources
- Local files (CSV, NPZ, HDF5)
- Potential: Database backends, API endpoints, public repositories

### Export Formats
- JSON (experiment metadata, conversations)
- CSV (analysis results, timeseries)
- NumPy (processed data arrays)

## Error Handling & Validation

### Data Validation
- File format verification
- Data shape consistency
- Type checking

### API Error Handling
- Rate limiting (exponential backoff)
- Timeout handling
- Graceful degradation (cache fallback)

### User Input Validation
- Required fields checking
- Parameter bounds validation
- Organism/technique vocabulary validation

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|-----------|---|
| Firing rate (1000 spikes) | O(n) | <1ms |
| Cross-correlation (1000 pts) | O(n log n) | ~10ms |
| Population sync (100 neurons) | O(n²) | ~100ms |
| LLM call | Network-bound | 2-10 sec |

## Security Considerations

- API keys stored in .env (not in code)
- Secrets not logged
- Input sanitization for file paths
- Rate limiting to prevent API abuse
- Optional data validation

## Extensibility Points

### Adding New LLM Providers
```python
class MyLLMAdapter(BaseLLMAdapter):
    def generate_response(self, prompt, **kwargs):
        # Implementation
        pass
```

### Adding New Analysis Methods
```python
class AnalysisTools:
    @staticmethod
    def new_analysis_method(data):
        # Implementation
        pass
```

### Adding New Research Tasks
```python
class ResearchTask(Enum):
    NEW_TASK = "new_task"
    
# Add prompt builder method
def build_new_task_prompt(...):
    # Implementation
    pass
```

## Testing Strategy

### Unit Tests (To Add)
- Individual analysis functions
- Data handler operations
- Metadata creation

### Integration Tests (To Add)
- End-to-end workflows
- LLM integration
- Data loading/saving

### Validation Tests (To Add)
- Configuration validation
- Input sanitation
- Error handling

## Future Development Roadmap

### Phase 1 (Current)
- ✅ Core analysis tools
- ✅ LLM integration
- ✅ Basic workflows

### Phase 2
- [ ] Database backend for experiments
- [ ] Web UI for experiment management
- [ ] Real-time data streaming
- [ ] Advanced ML methods

### Phase 3
- [ ] Integration with public datasets (Allen Institute, OpenNeuro)
- [ ] Multi-modal LLM (analyze images, videos)
- [ ] Collaborative features
- [ ] Publication pipeline

## Code Quality Standards

- **Documentation**: Comprehensive docstrings for all public methods
- **Type Hints**: Python 3.8+ type annotations throughout
- **Error Handling**: Explicit error handling with informative messages
- **Logging**: Structured logging for debugging and monitoring
- **Testing**: Aim for >80% code coverage

## Known Limitations

1. **LLM Integration**: Requires OpenAI API key (subscription required)
2. **Scalability**: Large datasets (>10GB) benefit from HDF5 + streaming
3. **Real-time**: Not optimized for real-time analysis (batch processing focus)
4. **Parallelization**: Limited multiprocessing support (future feature)

## File Organization Best Practices

```
project/
├── data/               # Raw experimental data
│   ├── exp_001/
│   │   ├── recording.npz
│   │   └── metadata.json
│   └── exp_002/
├── outputs/            # Analysis results
│   ├── figures/
│   ├── analyses/
│   └── exports/
├── logs/               # Log files
└── notebooks/          # Jupyter notebooks for exploration
```

## Performance Optimization Tips

1. **Data Loading**: Use NPZ for repeated access, HDF5 for large files
2. **Analysis**: Pre-compute frequently needed metrics
3. **LLM Calls**: Cache responses, batch requests when possible
4. **Visualization**: Prepare data offline, render separately
5. **Storage**: Archive old experiments, maintain active working set

---

## Quick Reference

### Key Classes
- `NeuroscienceRnDAssistant` - Main orchestrator
- `NeuroscienceRnDClient` - LLM interface
- `AnalysisTools` - Neuroscience analysis
- `ExperimentMetadata` - Experiment information
- `KnowledgeBase` - Domain knowledge storage

### Key Methods
- `create_experiment()` - Start new experiment
- `analyze_data()` - Get LLM analysis
- `design_experiment()` - Get experimental design
- `generate_hypotheses()` - Generate testable hypotheses
- `compute_firing_rate()` - Basic analysis

### Key Workflows
- Experiment Design → Data Collection → Analysis → Interpretation
- Observation → Hypothesis → Experiment → Results → Publication

---

**Document Version**: 1.0
**Last Updated**: February 2026
