import json
import logging
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import os


class AdaptiqLogger(logging.Handler):
    """
    A centralized logging handler for capturing and storing structured execution logs in-memory.

    This enhanced logger is designed to be the single point of logging for your entire system:
    - Acts as a singleton to ensure centralized log management
    - Collects all log messages (INFO, DEBUG, ERROR, etc.) in a thread-safe list
    - Automatically captures context information (module, function, line number)
    - Stores each log entry with timestamp, log level, description, context, and optional payload
    - Provides comprehensive methods for log retrieval, filtering, and management
    - Easily integrates with the Python logging system
    - Supports custom callbacks and structured payloads
    - Includes log rotation and size management capabilities

    Usage:
        # Initialize the centralized logger (typically done once at app startup)
        logger = AdaptiqLogger.get_instance()
        
        # Or use the setup method for quick initialization
        logger = AdaptiqLogger.setup(level=logging.INFO, max_logs=10000)

        # In any file, just use standard logging - it will be captured automatically
        import logging
        logging.info("User action performed", extra={"payload": {"user_id": 123, "action": "login"}})
        logging.error("Database connection failed", extra={"payload": {"db": "primary", "retry_count": 3}})

        # Retrieve logs
        all_logs = logger.get_logs()
        error_logs = logger.get_logs_by_type("ERROR")
        recent_logs = logger.get_recent_logs(limit=50)
    """

    _instance: Optional['AdaptiqLogger'] = None
    _lock_instance = Lock()

    def __init__(self, level=logging.NOTSET, max_logs: int = 10000, auto_rotate: bool = True):
        super().__init__(level)
        self.execution_logs: List[Dict[str, Any]] = []
        self._lock = Lock()  # Thread safety for the logs list
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []  # User callbacks
        self.max_logs = max_logs  # Maximum number of logs to keep in memory
        self.auto_rotate = auto_rotate  # Whether to automatically remove old logs
        self._total_logs_count = 0  # Track total logs ever created
        
        # Statistics tracking
        self._stats = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0
        }

    @classmethod
    def get_instance(cls) -> 'AdaptiqLogger':
        """
        Returns the singleton instance of AdaptiqLogger.
        Creates one if it doesn't exist.
        """
        if cls._instance is None:
            with cls._lock_instance:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def setup(cls, level=logging.INFO, max_logs: int = 10000, auto_rotate: bool = True) -> "AdaptiqLogger":
        """
        Class method to set up the centralized logger and attach it to the root logger.
        Returns the singleton handler instance.
        
        Args:
            level: Minimum logging level to capture
            max_logs: Maximum number of logs to keep in memory
            auto_rotate: Whether to automatically remove old logs when max is reached
        """
        # Get or create the singleton instance
        handler = cls.get_instance()
        handler.max_logs = max_logs
        handler.auto_rotate = auto_rotate
        handler.setLevel(level)

        # Get the root logger and add our handler if not already added
        root_logger = logging.getLogger()
        
        # Check if our handler is already attached
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)

        # Set the root logger level to capture all messages
        if root_logger.level > level:
            root_logger.setLevel(level)

        return handler

    def _get_caller_context(self) -> Dict[str, str]:
        """
        Extracts context information about where the log was called from.
        """
        try:
            # Go up the stack to find the actual caller (skip logging framework calls)
            frame = inspect.currentframe()
            for _ in range(10):  # Look up to 10 frames up
                frame = frame.f_back
                if frame is None:
                    break
                
                filename = frame.f_code.co_filename
                # Skip internal logging and our own files
                if ('logging' not in filename.lower() and 
                    'adaptiq' not in os.path.basename(filename).lower()):
                    return {
                        "module": os.path.basename(filename),
                        "function": frame.f_code.co_name,
                        "line": str(frame.f_lineno),
                        "file_path": filename
                    }
            
            # Fallback if we can't find a good frame
            return {
                "module": "unknown",
                "function": "unknown", 
                "line": "0",
                "file_path": "unknown"
            }
        except:
            return {
                "module": "unknown",
                "function": "unknown",
                "line": "0", 
                "file_path": "unknown"
            }

    def _rotate_logs_if_needed(self) -> None:
        """
        Removes old logs if we've exceeded the maximum count and auto_rotate is enabled.
        """
        if self.auto_rotate and len(self.execution_logs) >= self.max_logs:
            # Remove the oldest 20% of logs to make room
            remove_count = int(self.max_logs * 0.2)
            self.execution_logs = self.execution_logs[remove_count:]

    def emit(self, record: logging.LogRecord) -> None:
        """
        Called whenever a log message is emitted. Captures the log record,
        stores it in our internal structure, and notifies any registered callbacks.
        """
        try:
            # Get the current timestamp when this log is actually being processed
            timestamp = datetime.now().isoformat()

            # Get caller context information
            context = self._get_caller_context()

            # Support user-passed structured payload via `extra={"payload": ...}`
            payload = record.__dict__.get("payload", {})

            # Create the enhanced log entry
            log_entry = {
                "id": self._total_logs_count + 1,
                "timestamp": timestamp,
                "type": record.levelname,
                "description": record.getMessage(),
                "context": context,
                "payload": payload,
                "logger_name": record.name,
                "thread_name": getattr(record, 'threadName', 'MainThread')
            }

            # Thread-safe operations
            with self._lock:
                self._total_logs_count += 1
                self._stats[record.levelname] = self._stats.get(record.levelname, 0) + 1
                self.execution_logs.append(log_entry)
                self._rotate_logs_if_needed()

            # Invoke all registered callbacks
            for callback in self._callbacks:
                try:
                    callback(log_entry)
                except Exception as e:
                    # If a user callback throws, we don't want to break logging
                    pass  # Silently continue to avoid infinite loops

        except Exception:
            # If there's an error in our handler, we don't want to break the application
            self.handleError(record)

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Registers a callback function to be called on every new log entry.
        Each callback receives the structured log dictionary.
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Removes a previously registered callback.
        Returns True if the callback was found and removed, False otherwise.
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Returns all captured logs.
        """
        with self._lock:
            return self.execution_logs.copy()

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Returns the most recent N logs.
        """
        with self._lock:
            return self.execution_logs[-limit:] if limit > 0 else self.execution_logs.copy()

    def get_logs_json(self, pretty: bool = True) -> str:
        """
        Returns all captured logs as a JSON string.
        """
        indent = 2 if pretty else None
        return json.dumps(self.get_logs(), indent=indent)

    def get_logs_by_type(self, log_type: str) -> List[Dict[str, Any]]:
        """
        Returns logs filtered by type (INFO, DEBUG, ERROR, WARNING, CRITICAL).
        """
        log_type = log_type.upper()
        with self._lock:
            return [log for log in self.execution_logs if log["type"] == log_type]

    def get_logs_by_module(self, module_name: str) -> List[Dict[str, Any]]:
        """
        Returns logs filtered by the module (file) they originated from.
        """
        with self._lock:
            return [
                log for log in self.execution_logs 
                if log["context"]["module"] == module_name
            ]

    def get_logs_by_function(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Returns logs filtered by the function they originated from.
        """
        with self._lock:
            return [
                log for log in self.execution_logs 
                if log["context"]["function"] == function_name
            ]

    def get_logs_since(self, since_timestamp: Union[str, datetime]) -> List[Dict[str, Any]]:
        """
        Returns logs created after the specified timestamp.
        """
        if isinstance(since_timestamp, str):
            since_timestamp = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
        
        since_iso = since_timestamp.isoformat()
        
        with self._lock:
            return [
                log for log in self.execution_logs 
                if log["timestamp"] > since_iso
            ]

    def search_logs(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Searches for logs containing the specified text in their description.
        """
        if not case_sensitive:
            query = query.lower()
        
        with self._lock:
            results = []
            for log in self.execution_logs:
                description = log["description"]
                if not case_sensitive:
                    description = description.lower()
                if query in description:
                    results.append(log)
            return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the captured logs.
        """
        with self._lock:
            return {
                "total_logs_ever": self._total_logs_count,
                "current_logs_count": len(self.execution_logs),
                "max_logs_limit": self.max_logs,
                "auto_rotate_enabled": self.auto_rotate,
                "logs_by_type": self._stats.copy(),
                "callbacks_registered": len(self._callbacks)
            }

    def clear_logs(self) -> None:
        """
        Clears all captured logs and resets statistics.
        """
        with self._lock:
            self.execution_logs.clear()
            self._stats = {key: 0 for key in self._stats}

    def get_logs_count(self) -> int:
        """
        Returns the current number of captured logs in memory.
        """
        with self._lock:
            return len(self.execution_logs)

    def export_logs(self, filename: str, format: str = "json") -> bool:
        """
        Exports logs to a file in the specified format.
        
        Args:
            filename: Path to the output file
            format: Export format ("json" or "csv")
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            logs = self.get_logs()
            
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(logs, f, indent=2)
            elif format.lower() == "csv":
                import csv
                if logs:
                    with open(filename, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                        writer.writeheader()
                        for log in logs:
                            # Flatten complex fields for CSV
                            flattened_log = log.copy()
                            flattened_log['context'] = json.dumps(log['context'])
                            flattened_log['payload'] = json.dumps(log['payload'])
                            writer.writerow(flattened_log)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            return True
        except Exception:
            return False

    def __str__(self) -> str:
        """
        String representation showing current statistics.
        """
        stats = self.get_statistics()
        return (f"AdaptiqLogger(logs: {stats['current_logs_count']}/{stats['max_logs_limit']}, "
                f"total_ever: {stats['total_logs_ever']}, "
                f"callbacks: {stats['callbacks_registered']})")

    def __repr__(self) -> str:
        return self.__str__()


# Convenience functions for easy access to the centralized logger
def get_logger() -> AdaptiqLogger:
    """Convenience function to get the singleton logger instance."""
    return AdaptiqLogger.get_instance()

def setup_centralized_logging(level=logging.INFO, max_logs: int = 10000, auto_rotate: bool = True) -> AdaptiqLogger:
    """Convenience function to set up centralized logging for the entire application."""
    return AdaptiqLogger.setup(level=level, max_logs=max_logs, auto_rotate=auto_rotate)