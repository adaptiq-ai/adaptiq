from abc import ABC, abstractmethod
from typing import Optional

class BaseInstrumental(ABC):
    """
    Abstract base class for instrumentation.
    Defines the abstract 'run' method that concrete instrumental classes must implement.
    """

    @abstractmethod
    def run(self, config_path: Optional[str] = None, enable_pipeline: bool = True, prompt_auto_update: bool = False, feedback: Optional[str] = None):
        """
        Abstract method to instrument a function with execution timing and optional AdaptiQ pipeline triggering.
        Concrete implementations must provide the logic for this method.

        Args:
            config_path (str, optional): Path to the adaptiq_config.yml file. If None, uses default path.
            enable_pipeline (bool, optional): Whether to trigger the AdaptiQ pipeline. Defaults to True.
            prompt_auto_update (bool, optional): Whether to enable prompt auto-update. Defaults to False.
            feedback (str, optional): Human feedback about agent performance for prompt optimization. Defaults to None.
        """
        pass # No implementation in the abstract class