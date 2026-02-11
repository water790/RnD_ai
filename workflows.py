"""
Example workflows demonstrating the neuroscience R&D assistant with LLM integration
"""

import sys
from pathlib import Path

# Import the modules
from main import (
    NeuroscienceRnDAssistant,
    DataHandler,
    AnalysisTools,
    ExperimentDesigner,
    KnowledgeBase,
    create_llm_context
)
from llm_integration import (
    GPTAdapter,
    NeuroscienceRnDClient,
    ResearchTask,
    NeurosciencePromptBuilder
)
from visualization import NeuroscienceVisualizations, AnalysisVisualizer
import numpy as np


def workflow_experiment_design():
    """Workflow: Design a new experiment with LLM assistance"""
    print("="*80)
    print("WORKFLOW: Experiment Design with LLM Assistance")
    print("="*80)
    
    # Initialize components
    assistant = NeuroscienceRnDAssistant(model="gpt-4")
    gpt_adapter = GPTAdapter(model="gpt-4")
    llm_client = NeuroscienceRnDClient(gpt_adapter)
    
    # Step 1: Create experiment record
    print("\n1. Creating experiment record...")
    exp = assistant.create_experiment(
        experiment_id="EXP_002_V1_Motion",
        title="Motion tuning in V1 layer 2/3",
        organism="Mus musculus",
        brain_region="Primary Visual Cortex (V1), Layer 2/3",
        technique="In vivo two-photon calcium imaging",
        description="Investigation of motion direction selectivity in V1 circuits",
        pi="Dr. Jane Smith"
    )
    print(f"   Created: {exp.title}")
    
    # Step 2: Get experimental design from LLM
    print("\n2. Requesting experimental design from LLM...")
    design_recommendation = llm_client.design_experiment(
        background=f"Visual motion processing in rodent cortex, focusing on {exp.brain_region}",
        objective="Design an imaging experiment to characterize direction selectivity and motion adaptation"
    )
    print("   LLM Recommendation:")
    print("   " + "\n   ".join(design_recommendation.split("\n")[:10]))
    
    # Step 3: Get sample size recommendation
    print("\n3. Computing sample size recommendations...")
    sample_size_rec = ExperimentDesigner.suggest_sample_size(
        effect_size=0.7,
        alpha=0.05,
        power=0.8,
        test_type="paired t-test"
    )
    print(f"   Suggested n per group: {sample_size_rec['suggested_n_per_group']}")
    print(f"   Total sample size: {sample_size_rec['total_n']}")
    
    # Step 4: Create experimental plan
    print("\n4. Creating detailed experimental plan...")
    plan = ExperimentDesigner.create_experimental_plan(
        hypothesis="Layer 2/3 neurons maintain direction selectivity across motion speeds",
        organism="Mus musculus",
        brain_region="V1 Layer 2/3",
        expected_duration_hours=4.0,
        n_animals=5
    )
    print(f"   Plan duration: {plan['expected_duration_hours']} hours")
    print(f"   Sample size: {plan['n_animals']} animals")
    
    return assistant, llm_client


def workflow_data_analysis():
    """Workflow: Analyze experimental data with LLM interpretation"""
    print("\n" + "="*80)
    print("WORKFLOW: Data Analysis and Interpretation")
    print("="*80)
    
    assistant = NeuroscienceRnDAssistant()
    gpt_adapter = GPTAdapter(model="gpt-4")
    llm_client = NeuroscienceRnDClient(gpt_adapter)
    
    # Create experiment
    print("\n1. Setting up analysis experiment...")
    exp = assistant.create_experiment(
        experiment_id="EXP_003_DataAnalysis",
        title="Population recording analysis",
        organism="Mus musculus",
        brain_region="Primary Motor Cortex (M1)",
        technique="High-density electrophysiology",
        pi="Dr. John Doe"
    )
    
    # Simulate some neural data
    print("\n2. Generating simulated neural data...")
    
    # Simulate spike raster: 20 neurons, 1000 timepoints
    spike_matrix = np.random.binomial(1, 0.01, size=(20, 1000))
    
    # Simulate trial structure
    trial_starts = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    
    # Compute metrics
    print("\n3. Computing population metrics...")
    metrics = AnalysisTools.compute_raster_metrics(spike_matrix)
    
    print(f"   Total spikes: {metrics['total_spikes']}")
    print(f"   Mean firing rate: {metrics['mean_firing_rate']:.2f} Hz")
    print(f"   Population synchrony: {metrics['population_synchrony']:.3f}")
    
    # Compute firing rates for a single neuron
    spike_times = np.where(spike_matrix[0, :])[0].astype(float) * 0.001  # Convert to seconds
    firing_rates, bins = AnalysisTools.compute_firing_rate(spike_times, window_size=0.05)
    
    print(f"   Neuron 0 firing rate range: {firing_rates.min():.1f} - {firing_rates.max():.1f} Hz")
    
    # Get LLM interpretation
    print("\n4. Requesting LLM interpretation of results...")
    data_summary = f"""
    Population Recording Data:
    - {metrics['n_neurons']} neurons recorded
    - {metrics['n_timepoints']} timepoints
    - Total spikes: {metrics['total_spikes']}
    - Mean firing rate: {metrics['mean_firing_rate']:.2f} Hz
    - Population synchrony: {metrics['population_synchrony']:.3f}
    
    Single neuron firing rate varied between {firing_rates.min():.1f} and {firing_rates.max():.1f} Hz
    """
    
    interpretation = llm_client.interpret_results(
        experiment_context=f"Experiment: {exp.title}. Recording from {exp.brain_region}",
        results=data_summary,
        specific_question="What can we infer about population coding from these metrics?"
    )
    
    print("   LLM Interpretation:")
    print("   " + "\n   ".join(interpretation.split("\n")[:8]))
    
    return assistant, llm_client, metrics


def workflow_hypothesis_generation():
    """Workflow: Generate new hypotheses from observations"""
    print("\n" + "="*80)
    print("WORKFLOW: Hypothesis Generation from Data")
    print("="*80)
    
    gpt_adapter = GPTAdapter(model="gpt-4")
    llm_client = NeuroscienceRnDClient(gpt_adapter)
    
    print("\n1. Setting up hypothesis generation scenario...")
    
    observation = """
    Unexpected finding: Motor cortex neurons show stronger synchronization during
    motor preparation compared to movement execution, contrary to our initial hypothesis
    that synchronization should be highest during active movement.
    """
    
    print(f"   Observation: {observation}")
    
    print("\n2. Requesting hypothesis generation from LLM...")
    hypotheses = llm_client.generate_hypotheses(
        background="Motor cortex (M1) contains neurons involved in both motor planning and execution",
        observation=observation,
        focus="What mechanisms might explain stronger synchronization during preparation than execution?"
    )
    
    print("   Generated Hypotheses:")
    print("   " + "\n   ".join(hypotheses.split("\n")[:12]))
    
    return llm_client


def workflow_knowledge_base():
    """Workflow: Build and use a knowledge base"""
    print("\n" + "="*80)
    print("WORKFLOW: Knowledge Base Management")
    print("="*80)
    
    kb = KnowledgeBase()
    
    print("\n1. Adding entries to knowledge base...")
    
    # Add entries about different techniques
    kb.add_entry(
        key="technique_two_photon",
        content="Two-photon microscopy allows imaging of neural activity in vivo...",
        tags=["imaging", "techniques", "calcium"],
        metadata={"type": "technique"}
    )
    
    kb.add_entry(
        key="technique_ephys",
        content="Electrophysiology provides high temporal resolution recording...",
        tags=["recording", "techniques", "spikes"],
        metadata={"type": "technique"}
    )
    
    kb.add_entry(
        key="analysis_pca",
        content="Principal component analysis for dimensionality reduction...",
        tags=["analysis", "statistics", "reduction"],
        metadata={"type": "analysis_method"}
    )
    
    print(f"   Added 3 entries to knowledge base")
    
    print("\n2. Searching by tag...")
    techniques = kb.search_by_tag("techniques")
    print(f"   Found {len(techniques)} technique entries: {techniques}")
    
    print("\n3. Retrieving specific entry...")
    entry = kb.get_entry("analysis_pca")
    if entry:
        print(f"   Retrieved: {entry['content'][:50]}...")
    
    return kb


def workflow_visualization_preparation():
    """Workflow: Prepare visualization data"""
    print("\n" + "="*80)
    print("WORKFLOW: Visualization Data Preparation")
    print("="*80)
    
    viz = NeuroscienceVisualizations()
    visualizer = AnalysisVisualizer()
    
    print("\n1. Preparing spike raster data...")
    # Simulate 10 neurons with spikes
    spike_times_list = [np.random.uniform(0, 10, size=np.random.randint(5, 20)) for _ in range(10)]
    raster_data = viz.spike_raster_data(spike_times_list)
    print(f"   Created raster for {len(raster_data['neurons'])} neurons")
    print(f"   Total spikes: {sum(raster_data['spike_counts'])}")
    
    print("\n2. Preparing heatmap data...")
    neural_activity = np.random.randn(30, 100)
    heatmap_data = viz.heatmap_data(neural_activity, rows_label="Neurons", cols_label="Time (ms)")
    print(f"   Heatmap shape: {heatmap_data['shape']}")
    print(f"   Value range: {heatmap_data['original_range']}")
    
    print("\n3. Preparing tuning curve data...")
    orientations = np.linspace(0, 180, 13)
    responses = 50 * np.exp(-((orientations - 90)**2) / (2 * 30**2)) + 10
    tuning_data = viz.tuning_curve_data(orientations, responses)
    print(f"   Peak response: {tuning_data['peak_response']:.1f} Hz at {tuning_data['peak_stimulus']:.0f}°")
    
    print("\n4. Creating figure specification...")
    figure = visualizer.create_analysis_figure(
        figure_type='tuning_curve',
        data=tuning_data,
        title='Orientation Tuning Curve',
        figure_size=(10, 6)
    )
    print(f"   Created figure: {figure['title']}")
    
    return viz, visualizer


def workflow_full_pipeline():
    """Complete pipeline: Design → Analyze → Interpret → Generate new hypotheses"""
    print("\n" + "="*80)
    print("WORKFLOW: Full Research Pipeline")
    print("="*80)
    
    assistant = NeuroscienceRnDAssistant()
    gpt_adapter = GPTAdapter(model="gpt-4")
    llm_client = NeuroscienceRnDClient(gpt_adapter)
    
    print("\n PHASE 1: Experimental Design")
    print("-" * 40)
    exp = assistant.create_experiment(
        experiment_id="FULL_PIPELINE_001",
        title="Spatiotemporal dynamics of visual attention in V1",
        organism="Mus musculus",
        brain_region="Primary Visual Cortex",
        technique="Two-photon calcium imaging + optogenetics",
        pi="Dr. Research Lead",
        description="Study how attention modulates spatial and temporal properties of V1 responses"
    )
    print(f"Created: {exp.title}")
    
    print("\n PHASE 2: Data Simulation and Analysis")
    print("-" * 40)
    spike_matrix = np.random.binomial(1, 0.015, size=(50, 500))
    metrics = AnalysisTools.compute_raster_metrics(spike_matrix)
    print(f"Simulated population: {metrics['n_neurons']} neurons, {metrics['total_spikes']} total spikes")
    
    print("\n PHASE 3: LLM-Assisted Analysis")
    print("-" * 40)
    context = create_llm_context(assistant, exp.experiment_id, "attention_modulation_analysis")
    print(f"Generated LLM context ({len(context)} characters)")
    
    print("\n PHASE 4: Results Interpretation")
    print("-" * 40)
    results_summary = f"""
    - Population increased synchrony during attention task
    - Firing rate modulation ranged from -20% to +40%
    - Temporal receptive fields shifted earlier during attention
    """
    interpretation = llm_client.interpret_results(
        experiment_context=f"{exp.title} in {exp.brain_region}",
        results=results_summary
    )
    print("Interpretation generated (first 300 chars):")
    print(interpretation[:300] + "...")
    
    print("\n PHASE 5: Future Directions")
    print("-" * 40)
    hypotheses = llm_client.generate_hypotheses(
        background=exp.description,
        observation="Attention causes earlier temporal response shifts in V1"
    )
    print("Generated hypotheses for follow-up studies")
    
    return assistant, llm_client


if __name__ == "__main__":
    print("\nNEUROSCIENCE R&D ASSISTANT - EXAMPLE WORKFLOWS")
    print("=" * 80)
    
    print("\nNote: Some workflows require OPENAI_API_KEY environment variable")
    print("Install requirements: pip install openai numpy\n")
    
    # Run workflows
    try:
        # These don't require API key
        print("\n[Running: Knowledge Base Workflow]")
        workflow_knowledge_base()
        
        print("\n[Running: Visualization Workflow]")
        workflow_visualization_preparation()
        
        # These require API key but gracefully handle missing credentials
        print("\n[Running: Experiment Design Workflow]")
        workflow_experiment_design()
        
        print("\n[Running: Data Analysis Workflow]")
        workflow_data_analysis()
        
        print("\n[Running: Hypothesis Generation Workflow]")
        workflow_hypothesis_generation()
        
        print("\n[Running: Full Pipeline Workflow]")
        workflow_full_pipeline()
        
    except Exception as e:
        print(f"\nNote: Some workflows require additional setup (e.g., API keys)")
        print(f"Error encountered: {e}")
    
    print("\n" + "="*80)
    print("Workflow examples completed!")
    print("="*80)
