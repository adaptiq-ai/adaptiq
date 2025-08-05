import functools
import json
import time
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import weakref
from threading import local
from adaptiq.core.entities.adaptiq_meterics import CallInfo, TokenStats

class TokenTracker:
    """Thread-safe token tracking system"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._stats = defaultdict(TokenStats)
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
    def track_tokens(self, mode: str, call_info: CallInfo):
        """Thread-safe method to track tokens"""
        with self._lock:
            self._stats[mode].add_call(call_info)
    
    def get_stats(self, mode: Optional[str] = None) -> Union[Dict[str, TokenStats], TokenStats]:
        """Get statistics for a specific mode or all modes"""
        with self._lock:
            if mode:
                return self._stats[mode]
            return dict(self._stats)
    
    def get_stats_dict(self, mode: Optional[str] = None) -> Dict:
        """Get statistics as dictionary for serialization"""
        with self._lock:
            if mode:
                return self._stats[mode].dict()
            return {
                mode: stats.dict()
                for mode, stats in self._stats.items()
            }
    
    def reset(self, mode: Optional[str] = None):
        """Reset statistics"""
        with self._lock:
            if mode:
                self._stats[mode] = TokenStats()
            else:
                self._stats.clear()
    
    def export_to_file(self, filepath: str):
        """Export statistics to JSON file"""
        with self._lock:
            stats_dict = self.get_stats_dict()
            stats_dict['session_id'] = self.session_id
            stats_dict['export_timestamp'] = datetime.now().isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(stats_dict, f, indent=2)


class ThreadLocalData(local):
    response_stack: List[List[str]] 

# Thread-local storage for response capture
_thread_local: ThreadLocalData = ThreadLocalData()


class ResponseCapture:
    """Thread-safe context manager to capture LLM responses"""

    def __init__(self):
        self.responses = []
        self.capture_id = str(uuid.uuid4())

    def __enter__(self):
        if not hasattr(_thread_local, 'response_stack'):
            _thread_local.response_stack = []
        _thread_local.response_stack.append(self.responses)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(_thread_local, 'response_stack') and _thread_local.response_stack:
            _thread_local.response_stack.pop()


def capture_llm_response(response):
    """Thread-safe function to capture LLM responses"""
    if (hasattr(_thread_local, 'response_stack') and 
        _thread_local.response_stack):
        current_responses: List = _thread_local.response_stack[-1]
        current_responses.append(response)
    return response


# Global tracker registry with weak references to avoid memory leaks
_tracker_registry = weakref.WeakValueDictionary()
_registry_lock = threading.Lock()


def get_tracker(session_id: Optional[str] = None) -> TokenTracker:
    """Get or create a TokenTracker instance"""
    if session_id is None:
        # Create a new tracker for this thread if none specified
        thread_id = threading.get_ident()
        session_id = f"thread_{thread_id}_{int(time.time())}"
    
    with _registry_lock:
        if session_id not in _tracker_registry:
            _tracker_registry[session_id] = TokenTracker(session_id)
        return _tracker_registry[session_id]


def instrumental_track_tokens(
    mode: str, 
    provider: str = "openai",
    tracker: Optional[TokenTracker] = None,
    session_id: Optional[str] = None
):
    """
    Enhanced thread-safe decorator to track tokens for functions using LangChain invoke() method.

    Args:
        mode: String identifier to group/aggregate results
        provider: Provider name (default: "openai")
        tracker: Optional TokenTracker instance. If None, uses/creates thread-local tracker
        session_id: Optional session ID for tracker identification
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create tracker
            token_tracker = tracker or get_tracker(session_id)
            
            # Validate provider
            provider_lower = provider.lower()
            if provider_lower != "openai":
                print(f"Warning: Provider '{provider}' not fully supported for token tracking.")
                return func(*args, **kwargs)

            thread_id = str(threading.get_ident())
            
            # Capture responses during function execution
            with ResponseCapture() as capture:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Extract token usage from captured responses
                total_input_tokens = 0
                total_output_tokens = 0
                total_tokens = 0

                if capture.responses:
                    for response in capture.responses:
                        try:
                            if (hasattr(response, "usage_metadata") and 
                                response.usage_metadata):
                                usage = response.usage_metadata
                                total_input_tokens += usage.get("input_tokens", 0)
                                total_output_tokens += usage.get("output_tokens", 0)
                                total_tokens += usage.get("total_tokens", 0)
                        except Exception as e:
                            print(f"Error extracting tokens from response: {e}")

                # Create call info with validation
                call_info = CallInfo(
                    function_name=func.__name__,
                    timestamp=datetime.now().isoformat(),
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=total_tokens,
                    llm_calls=len(capture.responses),
                    session_id=token_tracker.session_id,
                    thread_id=thread_id,
                    execution_time=execution_time
                )

                # Track tokens using thread-safe method
                token_tracker.track_tokens(mode, call_info)

                # Log results
                if total_tokens > 0:
                    print(
                        f"✅ {func.__name__} [{mode}] [Session: {token_tracker.session_id[:8]}...] [Thread: {thread_id}]: "
                        f"{total_tokens} tokens ({total_input_tokens} in, {total_output_tokens} out) "
                        f"from {len(capture.responses)} LLM calls in {execution_time:.2f}s"
                    )
                else:
                    print(f"Warning: Could not extract token usage from {func.__name__} result")

            return result

        return wrapper
    return decorator


# Convenience functions for common use cases
def create_session_tracker(session_id: Optional[str] = None) -> TokenTracker:
    """Create a new session tracker"""
    return get_tracker(session_id)


@contextmanager
def token_tracking_session(session_id: Optional[str] = None):
    """Context manager for token tracking sessions"""
    tracker = create_session_tracker(session_id)
    try:
        yield tracker
    finally:
        # Cleanup if needed
        pass


