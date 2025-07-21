# src/adaptiq/__init__.py

# --- Version of the adaptiq package ---

__version__ = "0.0.0"  # Or your current version


# --- Logging ---
try:
    from .logger.adaptiq_trace_logger import AdaptiqTraceLogger
except ImportError:
    pass

# --- Decorators ---
try:
    from .instrumental.instrumental import (
        get_token_stats,
        instrumental_agent_logger,
        instrumental_crew_logger,
        instrumental_run,
        instrumental_task_logger,
        instrumental_track_tokens,
    )
except ImportError:
    pass

# --- Wizard ---
try:
    from .wizard.assistant import (
        AdaptiqWizardAssistant,
        adaptiq_run_wizard,
        adaptiq_run_wizard_headless,
    )
except ImportError:
    pass

# --- Orchestration ---
try:
    # ADD THIS LINE TO EXPOSE THE PIPELINE FUNCTION:
    from .orchestrations.pre_run_orchestrator import (
        AdaptiqPreRunOrchestrator,
        adaptiq_pre_run_pipeline,
    )
except ImportError:
    pass

try:
    # ADD THIS LINE TO EXPOSE THE PIPELINE FUNCTION:
    from .orchestrations.post_run_orchestrator import (
        AdaptiqPostRunOrchestrator,
        adaptiq_post_run_pipeline,
    )
except ImportError:
    pass

try:
    # ADD THIS LINE TO EXPOSE THE PIPELINE FUNCTION:
    from .orchestrations.reconciliation_orchestrator import (
        AdaptiqReconciliationOrchestrator,
        adaptiq_reconciliation_pipeline,
    )
except ImportError:
    pass

# --- Parsing ---
try:
    from .parser.prompt_parser import AdaptiqPromptParser
except ImportError:
    pass

try:
    from .parser.logs_parser import AdaptiqLogParser
except ImportError:
    pass

# --- Q-Learning ---
try:
    from .q_learning.q_learning import AdaptiqOfflineLearner
except ImportError:
    pass

# --- Mapper ---
try:
    from .mapper.mapping import AdaptiqStateActionExtractor, AdaptiqStateMapper
except ImportError:
    pass

# --- Utilities (especially for pre-run phase) ---
try:
    from .utils.pre_run_utils import (
        AdaptiqHypotheticalStateGenerator,
        AdaptiqPromptConsulting,
    )
except ImportError:
    pass

try:
    from .utils.post_run_utils import AdaptiqAgentTracer, AdaptiqPostRunValidator
except ImportError:
    pass

try:
    from .utils.reconciliation_utils import (
        AdaptiqPromptEngineer,
        AdaptiqQtablePostrunUpdate,
    )
except ImportError:
    pass

# --- Aggregator ---
try:
    from .aggregator.aggregator import AdaptiqAggregator
except ImportError:
    pass


# --- Convenience function to get the package version ---
def get_version():
    return __version__


import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
