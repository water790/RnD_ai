# Configuration file for Neuroscience R&D Assistant
# Copy this to config.py and customize for your setup

import os
from pathlib import Path

# ============================================================================
# LLM Configuration
# ============================================================================

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4"  # Options: "gpt-4", "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7  # 0.0-1.0 (lower = more deterministic)
DEFAULT_MAX_TOKENS = 2000

# ============================================================================
# Data Configuration
# ============================================================================

# Data directories
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
LOGS_DIR = Path("logs")

# Supported data formats
SUPPORTED_DATA_FORMATS = [".csv", ".npz", ".h5", ".hdf5"]

# ============================================================================
# Analysis Configuration
# ============================================================================

# Firing rate computation
FIRING_RATE_WINDOW_DEFAULT = 0.1  # seconds

# Cross-correlation parameters
CROSS_CORRELATION_MAX_LAG = 100  # samples
CROSS_CORRELATION_STEP = 1

# Population analysis
MIN_NEURONS_FOR_SYNC = 2
MAX_SYNCHRONY_LAG = 50  # milliseconds

# ============================================================================
# Experiment Configuration
# ============================================================================

# Default organisms
DEFAULT_ORGANISM = "Mus musculus"
SUPPORTED_ORGANISMS = [
    "Mus musculus",
    "Rattus norvegicus",
    "Homo sapiens",
    "Drosophila melanogaster",
    "Caenorhabditis elegans",
    "Danio rerio"
]

# Supported techniques
SUPPORTED_TECHNIQUES = [
    "Electrophysiology",
    "Two-photon calcium imaging",
    "Widefield fluorescence imaging",
    "fMRI",
    "EEG",
    "MEG",
    "Optogenetics",
    "Patch-clamp",
    "Multi-electrode arrays (MEA)",
    "High-density probes",
    "Neuropixels",
]

# Supported brain regions (sample)
SUPPORTED_BRAIN_REGIONS = [
    "Primary Visual Cortex (V1)",
    "Primary Motor Cortex (M1)",
    "Prefrontal Cortex (PFC)",
    "Hippocampus",
    "Cerebellum",
    "Superior Colliculus",
    "Lateral Intraparietal Area (LIP)",
    "Posterior Parietal Cortex (PPC)",
    "Lateral Temporal Cortex",
    "Medial Temporal Lobe",
]

# ============================================================================
# Statistical Configuration
# ============================================================================

# Statistical testing parameters
DEFAULT_ALPHA = 0.05  # Significance level
DEFAULT_POWER = 0.8  # Statistical power (1 - beta)
DEFAULT_EFFECT_SIZE = 0.5  # Cohen's d

# Sample size calculation
MIN_SAMPLE_SIZE = 3
MAX_SAMPLE_SIZE = 1000

# ============================================================================
# Visualization Configuration
# ============================================================================

# Figure defaults
DEFAULT_FIGURE_SIZE = (12, 8)  # inches
DEFAULT_DPI = 100

# Color schemes
COLORMAP_ACTIVITY = "viridis"
COLORMAP_CONNECTIVITY = "RdBu_r"
COLOR_SPIKE = "black"
COLOR_STIMULUS = "red"

# ============================================================================
# Knowledge Base Configuration
# ============================================================================

# Knowledge base storage
KB_STORAGE_FORMAT = "json"  # or "database"
KB_AUTO_EXPORT = True
KB_EXPORT_INTERVAL = 3600  # seconds

# Default tags for organizing knowledge
DEFAULT_KB_TAGS = [
    "technique",
    "analysis_method",
    "protocol",
    "literature",
    "statistics",
    "troubleshooting",
    "theory",
    "tools",
]

# ============================================================================
# Workflow Configuration
# ============================================================================

# Experiment workflow steps
WORKFLOW_STEPS = [
    "design",
    "planning",
    "execution",
    "data_collection",
    "quality_control",
    "analysis",
    "interpretation",
    "publication",
]

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "neuroscience_rnd.log"

# ============================================================================
# Cache Configuration
# ============================================================================

# Cache LLM responses to save API calls
USE_RESPONSE_CACHE = True
CACHE_DIR = Path(".cache")
CACHE_EXPIRY = 7 * 24 * 3600  # 7 days in seconds

# ============================================================================
# Advanced Configuration
# ============================================================================

# Enable experimental features
ENABLE_BETA_FEATURES = False

# Parallel processing
USE_MULTIPROCESSING = False
NUM_WORKERS = 4

# Data validation
VALIDATE_DATA_ON_LOAD = True
STRICT_MODE = False  # Enforce strict data format requirements

# ============================================================================
# API Configuration
# ============================================================================

# API timeout settings
API_TIMEOUT = 30  # seconds
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1  # seconds

# Rate limiting
RATE_LIMIT_CALLS = 10
RATE_LIMIT_WINDOW = 60  # seconds

# ============================================================================
# Initialization
# ============================================================================

def initialize_config():
    """Create necessary directories and validate configuration"""
    for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. LLM features will not work.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")


# Auto-initialize when module is imported
initialize_config()
