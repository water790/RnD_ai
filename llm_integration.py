"""
LLM Integration module for neuroscience R&D assistant
Handles communication with GPT variants and specialized prompts
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class ResearchTask(Enum):
    """Types of research tasks the LLM can assist with"""
    DATA_ANALYSIS = "data_analysis"
    EXPERIMENTAL_DESIGN = "experimental_design"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    RESULT_INTERPRETATION = "result_interpretation"
    METHODOLOGY_REVIEW = "methodology_review"
    PUBLICATION_ASSISTANCE = "publication_assistance"


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters"""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        """Generate structured response (JSON format)"""
        pass


class GPTAdapter(BaseLLMAdapter):
    """Adapter for OpenAI GPT models"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize GPT adapter
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 2000
        
        if not self.api_key:
            logger.warning("No API key found. Please set OPENAI_API_KEY environment variable.")
    
    def generate_response(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate a response using GPT
        
        Note: Requires openai package: pip install openai
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            return "Error: openai package not installed"
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict,
        **kwargs
    ) -> Dict:
        """
        Generate structured (JSON) response
        
        Requires function calling capability (OpenAI API)
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed")
            return {}
        
        client = OpenAI(api_key=self.api_key)
        
        # Prepare function definition for function calling
        function_def = {
            "name": "output_analysis",
            "description": "Output analysis results in structured format",
            "parameters": schema
        }
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": function_def}],
            tool_choice={"type": "function", "function": {"name": "output_analysis"}},
            **kwargs
        )
        
        # Extract and parse the function call result
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)
        
        return {}


class NeurosciencePromptBuilder:
    """Build specialized prompts for neuroscience research"""
    
    @staticmethod
    def build_analysis_prompt(
        task: ResearchTask,
        experiment_context: str,
        data_summary: str,
        specific_question: str
    ) -> str:
        """Build a prompt for data analysis"""
        
        prompts = {
            ResearchTask.DATA_ANALYSIS: f"""
You are a neuroscience research assistant. Analyze the following experimental data and provide insights.

EXPERIMENT CONTEXT:
{experiment_context}

DATA SUMMARY:
{data_summary}

SPECIFIC QUESTION:
{specific_question}

Please provide:
1. Key findings from the data
2. Statistical significance considerations
3. Interpretation in context of neuroscience theory
4. Potential artifacts or confounds
5. Recommendations for further analysis

Be precise and cite relevant neuroscience principles.
""",
            
            ResearchTask.EXPERIMENTAL_DESIGN: f"""
You are a neuroscience research consultant. Help design this experiment:

CONTEXT:
{experiment_context}

QUESTION:
{specific_question}

Please provide:
1. Specific experimental hypotheses
2. Recommended methodology and techniques
3. Required sample sizes and statistical power
4. Controls and controls conditions
5. Data collection parameters and quality metrics
6. Potential confounds and how to address them
7. Timeline and resource requirements

Refer to current best practices in neuroscience research.
""",
            
            ResearchTask.RESULT_INTERPRETATION: f"""
You are a neuroscience data scientist. Interpret these results:

EXPERIMENT:
{experiment_context}

DATA/RESULTS:
{data_summary}

QUESTION:
{specific_question}

Please provide:
1. Summary of the findings
2. How results relate to existing literature
3. Mechanistic interpretation
4. Limitations of the current analysis
5. Implications for future research
6. Confidence assessment

Use appropriate neuroscience terminology and frameworks.
""",
            
            ResearchTask.HYPOTHESIS_GENERATION: f"""
You are a neuroscience researcher brainstorming new ideas. Generate hypotheses:

BACKGROUND:
{experiment_context}

OBSERVATION/DATA:
{data_summary}

FOCUS:
{specific_question}

Please provide:
1. 3-5 specific, testable hypotheses
2. Mechanisms that might explain the observations
3. Predictions for each hypothesis
4. How to distinguish between hypotheses experimentally
5. Potential implications

Make hypotheses falsifiable and grounded in neuroscience.
""",
        }
        
        return prompts.get(task, prompts[ResearchTask.DATA_ANALYSIS])
    
    @staticmethod
    def build_literature_prompt(
        research_topic: str,
        keywords: List[str],
        specific_focus: str = ""
    ) -> str:
        """Build a prompt for literature review assistance"""
        
        return f"""
You are a neuroscience literature expert. Synthesize research on this topic:

TOPIC: {research_topic}

KEY TERMS:
{", ".join(keywords)}

{f"SPECIFIC FOCUS: {specific_focus}" if specific_focus else ""}

Please provide:
1. Overview of the state of knowledge
2. Key theoretical frameworks
3. Important recent findings
4. Methodological approaches used
5. Outstanding questions and gaps
6. Emerging trends or controversies
7. Suggested reading (if you can cite specific papers)

Organize by subtopics if relevant. Be scientifically rigorous.
"""
    
    @staticmethod
    def build_methodology_review_prompt(
        methodology: str,
        experimental_setup: str,
        concerns: List[str] = None
    ) -> str:
        """Build a prompt for methodology review"""
        
        concerns_text = ""
        if concerns:
            concerns_text = f"\nSPECIFIC CONCERNS:\n- " + "\n- ".join(concerns)
        
        return f"""
You are an experienced neuroscience methodologist. Review this methodology:

TECHNIQUE/METHOD: {methodology}

EXPERIMENTAL SETUP:
{experimental_setup}
{concerns_text}

Please provide:
1. Strengths of this approach
2. Limitations and potential issues
3. Data quality considerations
4. Interpretation pitfalls to avoid
5. Best practices recommendations
6. How to validate results
7. Suitable control conditions

Be constructive and specific to this experimental context.
"""


class NeuroscienceRnDClient:
    """Client for interacting with LLM for neuroscience R&D"""
    
    def __init__(self, llm_adapter: BaseLLMAdapter):
        """
        Initialize the client
        
        Args:
            llm_adapter: Adapter for the LLM (e.g., GPTAdapter)
        """
        self.llm = llm_adapter
        self.conversation_history: List[Dict[str, str]] = []
        self.prompt_builder = NeurosciencePromptBuilder()
    
    def analyze_data(
        self,
        experiment_context: str,
        data_summary: str,
        question: str
    ) -> str:
        """Analyze experimental data with LLM assistance"""
        
        prompt = self.prompt_builder.build_analysis_prompt(
            task=ResearchTask.DATA_ANALYSIS,
            experiment_context=experiment_context,
            data_summary=data_summary,
            specific_question=question
        )
        
        response = self.llm.generate_response(prompt)
        self._add_to_history("data_analysis", prompt, response)
        return response
    
    def design_experiment(
        self,
        background: str,
        objective: str
    ) -> str:
        """Get assistance designing an experiment"""
        
        prompt = self.prompt_builder.build_analysis_prompt(
            task=ResearchTask.EXPERIMENTAL_DESIGN,
            experiment_context=background,
            data_summary="",
            specific_question=objective
        )
        
        response = self.llm.generate_response(prompt)
        self._add_to_history("experimental_design", prompt, response)
        return response
    
    def interpret_results(
        self,
        experiment_context: str,
        results: str,
        specific_question: str = ""
    ) -> str:
        """Interpret experimental results with LLM"""
        
        prompt = self.prompt_builder.build_analysis_prompt(
            task=ResearchTask.RESULT_INTERPRETATION,
            experiment_context=experiment_context,
            data_summary=results,
            specific_question=specific_question or "What do these results mean?"
        )
        
        response = self.llm.generate_response(prompt)
        self._add_to_history("result_interpretation", prompt, response)
        return response
    
    def generate_hypotheses(
        self,
        background: str,
        observation: str,
        focus: str = ""
    ) -> str:
        """Generate new hypotheses based on observations"""
        
        prompt = self.prompt_builder.build_analysis_prompt(
            task=ResearchTask.HYPOTHESIS_GENERATION,
            experiment_context=background,
            data_summary=observation,
            specific_question=focus or "What hypotheses might explain this?"
        )
        
        response = self.llm.generate_response(prompt)
        self._add_to_history("hypothesis_generation", prompt, response)
        return response
    
    def review_literature(
        self,
        topic: str,
        keywords: List[str],
        focus: str = ""
    ) -> str:
        """Get literature review assistance"""
        
        prompt = self.prompt_builder.build_literature_prompt(
            research_topic=topic,
            keywords=keywords,
            specific_focus=focus
        )
        
        response = self.llm.generate_response(prompt)
        self._add_to_history("literature_review", prompt, response)
        return response
    
    def review_methodology(
        self,
        methodology: str,
        setup: str,
        concerns: List[str] = None
    ) -> str:
        """Get methodology review and suggestions"""
        
        prompt = self.prompt_builder.build_methodology_review_prompt(
            methodology=methodology,
            experimental_setup=setup,
            concerns=concerns
        )
        
        response = self.llm.generate_response(prompt)
        self._add_to_history("methodology_review", prompt, response)
        return response
    
    def _add_to_history(self, task: str, prompt: str, response: str) -> None:
        """Add interaction to conversation history"""
        self.conversation_history.append({
            "task": task,
            "prompt": prompt,
            "response": response
        })
    
    def export_conversation(self, filepath: str) -> None:
        """Export conversation history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        logger.info(f"Exported conversation to {filepath}")


if __name__ == "__main__":
    # Example usage (requires OPENAI_API_KEY environment variable)
    
    # Initialize adapter and client
    gpt_adapter = GPTAdapter(model="gpt-4")
    client = NeuroscienceRnDClient(gpt_adapter)
    
    # Example: Design an experiment
    print("Requesting experiment design assistance...")
    design = client.design_experiment(
        background="Understanding visual processing in primary visual cortex",
        objective="Design a study to investigate how layer 2/3 neurons encode motion direction"
    )
    print(design)
    print("\n" + "="*80 + "\n")
    
    # Example: Generate hypotheses
    print("Generating hypotheses for an observation...")
    hypotheses = client.generate_hypotheses(
        background="Neurons in V1 show direction selectivity",
        observation="Some neurons respond equally to opposite directions when adapted",
        focus="What mechanisms might cause this adaptation?"
    )
    print(hypotheses)
