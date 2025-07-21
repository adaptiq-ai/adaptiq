import os
import logging
from dotenv import load_dotenv
from threading import Lock

logging.basicConfig(level=logging.INFO)

class Settings:
    """Singleton class to manage application settings."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._load_env()
        return cls._instance

    def _load_env(self):
        """Loads environment variables from a .env file."""
        load_dotenv(override=True, dotenv_path="./.env")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        logging.info("All settings loaded successfully.")
        

settings = Settings()