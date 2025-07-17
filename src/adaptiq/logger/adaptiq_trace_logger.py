import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from threading import Lock


class AdaptiqTraceLogger(logging.Handler):
    """
    A custom logging handler for capturing and storing structured execution logs in-memory.

    This logger is designed for the AdaptiQ framework to:
    - Collect all log messages (INFO, DEBUG, ERROR, etc.) in a thread-safe list.
    - Store each log entry with a timestamp, log level, description, and optional payload.
    - Provide methods to retrieve logs as a list or JSON, filter by type, count logs, and clear logs.
    - Easily integrate with the Python logging system via the `setup` class method.

    Usage:
        handler = AdaptiqTraceLogger.setup(level=logging.INFO)
        logging.info("This will be captured by AdaptiqTraceLogger.")
        logs = handler.get_logs()
    """
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.execution_logs: List[Dict[str, Any]] = []
        self._lock = Lock()  # Thread safety for the logs list
        
    def emit(self, record: logging.LogRecord) -> None:
        """
        Called whenever a log message is emitted. Captures the log record
        and stores it in our internal structure.
        """
        try:
            # Get the current timestamp when this log is actually being processed
            timestamp = datetime.now().isoformat()
            
            # Create the log entry
            log_entry = {
                "type": record.levelname,
                "description": record.getMessage(),
                "payload": {},  # Empty for now as requested
                "timestamp": timestamp
            }
            
            # Thread-safe append to the logs list
            with self._lock:
                self.execution_logs.append(log_entry)
                
        except Exception:
            # If there's an error in our handler, we don't want to break the application
            self.handleError(record)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Returns all captured logs in the requested format.
        """
        with self._lock:
            return self.execution_logs.copy()
    
    def get_logs_json(self) -> str:
        """
        Returns all captured logs as a JSON string.
        """
        return json.dumps(self.get_logs(), indent=2)
    
    def clear_logs(self) -> None:
        """
        Clears all captured logs.
        """
        with self._lock:
            self.execution_logs.clear()
    
    def get_logs_by_type(self, log_type: str) -> List[Dict[str, Any]]:
        """
        Returns logs filtered by type (INFO, DEBUG, ERROR, etc.).
        """
        with self._lock:
            return [log for log in self.execution_logs if log["type"] == log_type.upper()]
    
    def get_logs_count(self) -> int:
        """
        Returns the total number of captured logs.
        """
        with self._lock:
            return len(self.execution_logs)
    
    @classmethod
    def setup(cls, level=logging.INFO) -> 'AdaptiqTraceLogger':
        """
        Class method to easily set up the logger handler and attach it to the root logger.
        Returns the handler instance so you can call methods on it.
        Only captures INFO, WARNING, ERROR, and CRITICAL levels (excludes DEBUG).
        """
        # Create the handler
        handler = cls(level)
        
        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Set the root logger level to capture all messages
        if root_logger.level > level:
            root_logger.setLevel(level)
            
        return handler