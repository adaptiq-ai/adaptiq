import functools
import subprocess
import os
import yaml
import json
import time
import tracemalloc
from typing import Any, Callable, Dict, Optional, List
from datetime import datetime


# Global token tracking storage
_token_tracking = {}

# Global storage for capturing LLM responses
_captured_responses = []

# Global embeddings token tracking storage
_token_tracking_embeddings = {}

# Global counter to track crew executions
_crew_counter = 0

# Global variable to store all metrics
_crew_metrics: List[Dict[str, Any]] = []

class ResponseCapture:
    """Context manager to capture LLM responses"""
    def __init__(self):
        self.responses = []
        
    def __enter__(self):
        global _captured_responses
        _captured_responses = self.responses
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _captured_responses
        _captured_responses = []


def capture_llm_response(response):
    """Function to be called after each LLM invoke to capture the response"""
    global _captured_responses
    if _captured_responses is not None:
        _captured_responses.append(response)
    return response


def instrumental_run(config_path=None, enabled=True):
    """
    Decorator to instrument a function with execution timing and optional AdaptiQ pipeline triggering.

    - Prints the function name, start/end time, and result.
    - If trigger_command is True, runs the AdaptiQ pipeline via subprocess after the function completes,
      using the provided adaptiq_config.yml path.
    - If include_crew_metrics is True, captures crew metrics and includes them in the subprocess command.
    - Captures and displays the output in real-time from the subprocess.
    - Returns the subprocess output if triggered, otherwise returns the original function result.
    - Automatically searches for an existing 'results' folder to use as output path.
    - Handles alert mode logic to determine when to send reports.

    Args:
        config_path (str, optional): Path to the adaptiq_config.yml file. If None, uses default path.
        enabled (bool, optional): Whether to trigger the AdaptiQ pipeline. Defaults to True.
    """
    def find_results_folder():
        """Find the 'results' folder in current directory or parent directories."""
        current_dir = os.getcwd()
        
        # Check current directory first
        results_path = os.path.join(current_dir, 'results')
        if os.path.exists(results_path) and os.path.isdir(results_path):
            return results_path
        
        # Check parent directories up to 3 levels
        for i in range(3):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
            results_path = os.path.join(current_dir, 'results')
            if os.path.exists(results_path) and os.path.isdir(results_path):
                return results_path
        
        return None
    
    def get_alert_mode(config_path):
        """
        Reads the adaptiq_config.yml and extracts alert_mode settings.
        Returns:
            dict: {
                "mode": "on_demand" or "per_run" or "none",
                "runs": int or None
            }
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        alert_mode = config.get("alert_mode", {})
        if alert_mode.get("on_demand", {}).get("enabled", False):
            return {"mode": "on_demand", "runs": alert_mode["on_demand"].get("runs", None)}
        elif alert_mode.get("per_run", {}).get("enabled", False):
            return {"mode": "per_run", "runs": None}
        else:
            return {"mode": "none", "runs": None}
    
    def determine_should_send_report(crew_metrics, alert_mode_info):
        """
        Determines whether to send a report based on alert mode and crew metrics.
        
        Args:
            crew_metrics (list): List of crew metrics
            alert_mode_info (dict): Alert mode configuration
            
        Returns:
            bool: True if report should be sent, False otherwise
        """
        if not crew_metrics:
            return True  # Default to sending report if no crew metrics
        
        # Get current execution count from crew metrics
        current_execution_count = crew_metrics[-1].get('execution_count', 0) if crew_metrics else 0
        
        # Determine if we should send report based on alert mode
        if alert_mode_info["mode"] == "on_demand" and alert_mode_info["runs"]:
            # Send report only when we reach the target number of runs
            return current_execution_count >= alert_mode_info["runs"]
        elif alert_mode_info["mode"] == "per_run":
            # Send report after each run
            return True
        else:
            # Standard mode - send report
            return True
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            trigger_command = enabled
            include_crew_metrics = True
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Calculate execution time
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"[INSTRUMENT] Function {func.__name__} completed in {duration:.3f}s")
            print(f"[INSTRUMENT] Result: {result}")
            
            # Capture crew metrics if requested
            crew_metrics = None
            if include_crew_metrics:
                try:
                    # Import here to avoid circular imports
                    crew_metrics = get_crew_metrics()
                    
                    print("[INSTRUMENT] === CREW METRICS CAPTURED ===")
                    print(f"[INSTRUMENT] Total executions tracked: {len(crew_metrics)}")
                    
                    # Print summary of crew metrics for testing
                    if crew_metrics:
                        total_tokens = sum(metric.get('total_tokens', 0) for metric in crew_metrics)
                        total_time = sum(metric.get('execution_time_seconds', 0) for metric in crew_metrics)
                        print(f"[INSTRUMENT] Total tokens across all executions: {total_tokens:,}")
                        print(f"[INSTRUMENT] Total execution time: {total_time:.2f}s")
                        
                        # Show last execution details
                        if crew_metrics:
                            last_metric = crew_metrics[-1]
                            print(f"[INSTRUMENT] Last execution: {last_metric.get('execution_time_seconds', 0):.2f}s, "
                                  f"{last_metric.get('total_tokens', 0):,} tokens")
                    
                    print("[INSTRUMENT] === END CREW METRICS ===")
                    
                except ImportError:
                    print("[INSTRUMENT] Warning: Could not import get_crew_metrics function")
                    crew_metrics = None
                except Exception as e:
                    print(f"[INSTRUMENT] Warning: Error capturing crew metrics: {e}")
                    crew_metrics = None
            
            # Trigger command after function finishes
            if trigger_command:
                # Find the results folder
                output_path = find_results_folder()
                if output_path is None:
                    print("[INSTRUMENT] Warning: No 'results' folder found. Skipping AdaptiQ pipeline execution.")
                    return result
                
                print(f"[INSTRUMENT] Found results folder at: {output_path}")
                
                # Get alert mode configuration
                try:
                    alert_mode_info = get_alert_mode(config_path)
                    print(f"[INSTRUMENT] Alert mode detected: {alert_mode_info['mode']}")
                    if alert_mode_info['runs']:
                        print(f"[INSTRUMENT] Number of runs configured: {alert_mode_info['runs']}")
                except Exception as e:
                    print(f"[INSTRUMENT] Warning: Could not read alert mode configuration: {str(e)}")
                    alert_mode_info = {"mode": "none", "runs": None}
                
                # Determine if we should send report
                should_send_report = determine_should_send_report(crew_metrics, alert_mode_info)
                print(f"[INSTRUMENT] Should send report: {should_send_report}")
                
                config = config_path
                cmd_args = ["adaptiq", "run", "--config", config, "--output_path", output_path]
                
                # Add crew metrics as CLI argument if available
                if crew_metrics:
                    crew_metrics_json = json.dumps(crew_metrics)
                    cmd_args.extend(["--crew_metrics", crew_metrics_json])
                    print(f"[INSTRUMENT] Crew metrics added to command args (size: {len(crew_metrics_json)} chars)")
                
                # Add send_report flag
                cmd_args.extend(["--send_report", str(should_send_report).lower()])
                print(f"[INSTRUMENT] Send report flag added: {should_send_report}")
                
                # Set up environment
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                print(f"[INSTRUMENT] Triggering command: {' '.join(cmd_args[:4])}{'...' if len(cmd_args) > 4 else ''}")
                
                try:
                    # Start the process with real-time output capture
                    process = subprocess.Popen(
                        cmd_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Merge stderr with stdout
                        text=True,
                        encoding='utf-8',
                        errors='replace',  # Replace problematic characters
                        env=env,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )
                    
                    output_lines = []
                    
                    # Read output line by line and display in real-time
                    print("[INSTRUMENT] === ADAPTIQ PIPELINE OUTPUT ===")
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            # Clean the line
                            clean_line = line.rstrip()
                            
                            # Print to console
                            print(f"[ADAPTIQ] {clean_line}")
                            output_lines.append(clean_line)
                    
                    # Wait for process to complete
                    return_code = process.wait(timeout=600)  # 10 minute timeout
                    
                    print("[INSTRUMENT] === END ADAPTIQ PIPELINE OUTPUT ===")
                    
                    if return_code == 0:
                        print("[INSTRUMENT] AdaptiQ pipeline executed successfully")
                        # Return both subprocess output and crew metrics if requested
                        if include_crew_metrics and crew_metrics:
                            return {
                                'adaptiq_output': '\n'.join(output_lines),
                                'crew_metrics': crew_metrics,
                                'should_send_report': should_send_report
                            }
                        else:
                            return '\n'.join(output_lines)
                    else:
                        print(f"[INSTRUMENT] AdaptiQ pipeline failed with return code: {return_code}")
                        return None
                        
                except FileNotFoundError:
                    print("[INSTRUMENT] Error: 'adaptiq' command not found. Make sure AdaptiQ is installed and in PATH.")
                    return None
                    
                except subprocess.TimeoutExpired:
                    print("[INSTRUMENT] Error: AdaptiQ pipeline timed out after 10 minutes")
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except Exception:
                        pass
                    return None
                    
                except Exception as e:
                    print(f"[INSTRUMENT] Error executing AdaptiQ pipeline: {e}")
                    return None
            
            # Return the original result if no command triggered
            # But include crew metrics if requested and no subprocess was run
            if include_crew_metrics and crew_metrics and not trigger_command:
                return {
                    'original_result': result,
                    'crew_metrics': crew_metrics
                }
            
            return result
            
        return wrapper
    return decorator


def instrumental_agent_logger(func: Callable) -> Callable:
    """
    Decorator to automatically add step_callback logging to CrewAI agents.
    
    This decorator modifies the agent creation to include step_callback
    that logs agent steps/thoughts after each execution step.
    
    Args:
        func (callable): The function that creates and returns an Agent.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Import logger here to avoid circular imports and initialization issues
        from adaptiq.logger.adaptiq_crewai_logger import logger
        
        # Create a step callback function that logs thoughts
        def step_callback(step_output):
            """Callback to log agent steps/thoughts"""
            logger.log_thoughts(step_output)
        
        # Execute the original function to get the Agent
        agent = func(*args, **kwargs)
        
        # Add the step callback to the agent
        agent.step_callback = step_callback
        
        return agent
    
    return wrapper


def instrumental_task_logger(func: Callable) -> Callable:
    """
    Decorator to automatically add callback logging to CrewAI tasks.
    
    This decorator modifies the task creation to include callback
    that logs task information after task completion.
    
    Args:
        func (callable): The function that creates and returns a Task.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Import logger here to avoid circular imports and initialization issues
        from adaptiq.logger.adaptiq_crewai_logger import logger
        
        # Create a task callback function that logs task completion
        def task_callback(task_output):
            """Callback to log task completion"""
            logger.log_task(task_output)
        
        # Execute the original function to get the Task
        task = func(*args, **kwargs)
        
        # Add the callback to the task (CrewAI uses 'callback' not 'task_callback')
        task.callback = task_callback
        
        return task
    
    return wrapper


def instrumental_crew_logger(log_to_console: bool = True) -> Callable:
    """
    Decorator to track time, tokens, memory usage, model information, and execution count for CrewAI crew execution.

    This decorator can be applied to the crew kickoff method or any method that 
    executes a crew and returns a result with token_usage attribute.

    Args:
        log_to_console (bool): Whether to print metrics to console

    Usage:
        @instrumental_crew_logger(log_to_console=True)
        def run_crew(self):
            return self.crew().kickoff(inputs={"topic": "AI"})
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _crew_counter, _crew_metrics
            
            # Increment the crew counter
            _crew_counter += 1
            current_execution = _crew_counter
            
            # Start memory tracking
            tracemalloc.start()

            # Record start time
            start_time = time.time()
            start_timestamp = datetime.now()

            # Execute the original function
            result = func(*args, **kwargs)

            # Record end time
            end_time = time.time()
            end_timestamp = datetime.now()
            execution_time = end_time - start_time

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Extract token usage information
            token_usage = getattr(result, 'token_usage', None)

            # Extract model information from crew agents
            models_used = []
            
            # Store the crew instance in the result for access by the decorator
            if hasattr(result, '_crew_instance'):
                crew_instance = result._crew_instance
                if hasattr(crew_instance, 'agents'):
                    for agent in crew_instance.agents:
                        if hasattr(agent, 'llm') and hasattr(agent.llm, 'model'):
                            models_used.append({
                                'agent_role': getattr(agent, 'role', 'Unknown'),
                                'model': agent.llm.model
                            })

            # Initialize metrics
            metrics = {
                'execution_count': current_execution,
                'total_executions': _crew_counter,
                'start_timestamp': start_timestamp.isoformat(),
                'end_timestamp': end_timestamp.isoformat(),
                'execution_time_seconds': round(execution_time, 2),
                'execution_time_minutes': round(execution_time / 60, 2),
                'current_memory_mb': round(current / 1024 / 1024, 2),
                'peak_memory_mb': round(peak / 1024 / 1024, 2),
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cached_prompt_tokens': 0,
                'successful_requests': 0,
                'models_used': models_used,
                'function_name': func.__name__
            }

            # Process token usage if available
            if token_usage:
                metrics['total_tokens'] = getattr(token_usage, 'total_tokens', 0)
                metrics['prompt_tokens'] = getattr(token_usage, 'prompt_tokens', 0)
                metrics['completion_tokens'] = getattr(token_usage, 'completion_tokens', 0)
                metrics['cached_prompt_tokens'] = getattr(token_usage, 'cached_prompt_tokens', 0)
                metrics['successful_requests'] = getattr(token_usage, 'successful_requests', 0)

            # Store metrics in global variable
            _crew_metrics.append(metrics)

            # Log to console if requested
            if log_to_console:
                print("\n" + "="*50)
                print("🚀 CREW PERFORMANCE METRICS")
                print("="*50)
                print(f"🔢 Execution #{current_execution} (Total: {_crew_counter})")
                print(f"⏱️ Execution Time: {metrics['execution_time_seconds']}s ({metrics['execution_time_minutes']} min)")
                print(f"🧠 Current Memory: {metrics['current_memory_mb']} MB")
                print(f"📊 Peak Memory: {metrics['peak_memory_mb']} MB")
                print(f"🔢 Total Tokens: {metrics['total_tokens']:,}")
                print(f"📝 Prompt Tokens: {metrics['prompt_tokens']:,}")
                print(f"💾 Cached Prompt Tokens: {metrics['cached_prompt_tokens']:,}")
                print(f"✅ Completion Tokens: {metrics['completion_tokens']:,}")
                print(f"🔄 Successful Requests: {metrics['successful_requests']}")
                
                # Display model information
                if models_used:
                    print(f"🤖 Models Used:")
                    for model_info in models_used:
                        print(f"   • {model_info['agent_role']}: {model_info['model']}")
                else:
                    print("🤖 Models Used: Unable to detect")
                
                print("="*50 + "\n")

            return result

        return wrapper
    return decorator


def instrumental_track_tokens(mode: str, provider: str):
    """
    Enhanced decorator to track tokens for functions using LangChain invoke() method.
    
    Args:
        mode: String identifier to group/aggregate results (e.g., "analysis", "summarization")
        provider: Provider name (OpenAI only supported for now)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Read and validate config
            provider_lower = provider.lower()
                
            if provider_lower != 'openai':
                print(f"Warning: Provider '{provider}' not supported for token tracking. Only 'openai' is supported.")
                return func(*args, **kwargs)
            
            # Capture responses during function execution
            with ResponseCapture() as capture:
                result = func(*args, **kwargs)
                
                # Extract token usage from captured responses
                total_input_tokens = 0
                total_output_tokens = 0
                total_tokens = 0
                
                if capture.responses:
                    for response in capture.responses:
                        try:
                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                usage = response.usage_metadata
                                total_input_tokens += usage.get('input_tokens', 0)
                                total_output_tokens += usage.get('output_tokens', 0)
                                total_tokens += usage.get('total_tokens', 0)
                        except Exception as e:
                            print(f"Error extracting tokens from response: {e}")
                
                # Initialize mode tracking if not exists
                if mode not in _token_tracking:
                    _token_tracking[mode] = {
                        'total_input_tokens': 0,
                        'total_output_tokens': 0,
                        'total_tokens': 0,
                        'total_calls': 0,
                        'call_history': []
                    }
                
                # Update tracking for this mode
                _token_tracking[mode]['total_input_tokens'] += total_input_tokens
                _token_tracking[mode]['total_output_tokens'] += total_output_tokens
                _token_tracking[mode]['total_tokens'] += total_tokens
                _token_tracking[mode]['total_calls'] += 1
                
                # Add call details
                call_info = {
                    'function_name': func.__name__,
                    'timestamp': datetime.now().isoformat(),
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'total_tokens': total_tokens,
                    'llm_calls': len(capture.responses)
                }
                _token_tracking[mode]['call_history'].append(call_info)
                
                if total_tokens > 0:
                    print(f"✅ {func.__name__} [{mode}]: {total_tokens} tokens ({total_input_tokens} in, {total_output_tokens} out) from {len(capture.responses)} LLM calls")
                else:
                    print(f"Warning: Could not extract token usage from {func.__name__} result")
                
            return result
        
        return wrapper
    return decorator


def instrumental_track_embeddings(mode: str, provider: str):
    """
    Decorator to track embeddings usage for functions using embed_query() method.
    
    Args:
        mode: String identifier to group/aggregate results (e.g., "search", "similarity")
        provider: Provider name (e.g., "openai")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate provider
            provider_lower = provider.lower()
            
            if provider_lower not in ['openai']:
                print(f"Warning: Provider '{provider}' not supported for embeddings tracking. Supported: openai")
                return func(*args, **kwargs)
            
            # Initialize tracking for this function call
            call_embeddings_count = 0
            call_total_tokens = 0
            embedding_calls = []
            
            # Store original embed_query method to track calls
            original_embed_methods = {}
            
            # Function to wrap embed_query methods
            def create_embed_wrapper(original_method, obj_name="embeddings"):
                def embed_wrapper(*embed_args, **embed_kwargs):
                    nonlocal call_embeddings_count, call_total_tokens
                    
                    # Call original method
                    result = original_method(*embed_args, **embed_kwargs)
                    
                    # Track this embedding call
                    call_embeddings_count += 1
                    
                    # Estimate tokens (approximate: 1 token ≈ 4 characters for text)
                    if embed_args:
                        text_input = str(embed_args[0])
                        estimated_tokens = len(text_input) // 4 + 1  # Rough estimation
                    else:
                        estimated_tokens = 1  # Fallback
                    
                    call_total_tokens += estimated_tokens
                    
                    # Store embedding call details
                    embedding_call_info = {
                        'object_name': obj_name,
                        'input_text_length': len(str(embed_args[0])) if embed_args else 0,
                        'estimated_tokens': estimated_tokens,
                        'timestamp': datetime.now().isoformat()
                    }
                    embedding_calls.append(embedding_call_info)
                    
                    return result
                
                return embed_wrapper
            
            # Look for embedding objects in self (if it's a method call)
            if args and hasattr(args[0], '__dict__'):
                self_obj = args[0]
                # Find all potential embedding objects
                for attr_name, attr_value in self_obj.__dict__.items():
                    if hasattr(attr_value, 'embed_query'):
                        # Store original method
                        original_embed_methods[attr_name] = attr_value.embed_query
                        # Replace with wrapper using setattr to handle Pydantic models
                        try:
                            setattr(attr_value, 'embed_query', create_embed_wrapper(
                                attr_value.embed_query, 
                                attr_name
                            ))
                        except (ValueError, AttributeError):
                            # For Pydantic models that don't allow dynamic assignment,
                            # use monkey patching on the object itself
                            object.__setattr__(attr_value, 'embed_query', create_embed_wrapper(
                                attr_value.embed_query, 
                                attr_name
                            ))
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore original methods
                if args and hasattr(args[0], '__dict__'):
                    self_obj = args[0]
                    for attr_name, original_method in original_embed_methods.items():
                        if hasattr(self_obj, attr_name):
                            attr_value = getattr(self_obj, attr_name)
                            try:
                                setattr(attr_value, 'embed_query', original_method)
                            except (ValueError, AttributeError):
                                # Use monkey patching for restoration too
                                object.__setattr__(attr_value, 'embed_query', original_method)
            
            # Update tracking statistics
            try:
                # Initialize mode tracking if not exists
                if mode not in _token_tracking_embeddings:
                    _token_tracking_embeddings[mode] = {
                        'total_embeddings_count': 0,
                        'total_estimated_tokens': 0,
                        'total_calls': 0,
                        'call_history': []
                    }
                
                # Update tracking for this mode
                _token_tracking_embeddings[mode]['total_embeddings_count'] += call_embeddings_count
                _token_tracking_embeddings[mode]['total_estimated_tokens'] += call_total_tokens
                _token_tracking_embeddings[mode]['total_calls'] += 1
                
                # Add call details
                call_info = {
                    'function_name': func.__name__,
                    'provider': provider,
                    'timestamp': datetime.now().isoformat(),
                    'embeddings_count': call_embeddings_count,
                    'estimated_tokens': call_total_tokens,
                    'embedding_calls': embedding_calls
                }
                _token_tracking_embeddings[mode]['call_history'].append(call_info)
                
                print(f"✅ {func.__name__} [{mode}] ({provider}): {call_embeddings_count} embeddings, ~{call_total_tokens} tokens")
                
            except Exception as e:
                print(f"Error tracking embeddings for {func.__name__}: {e}")
            
            return result
        
        return wrapper
    return decorator


def get_token_stats(mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Get token statistics for a specific mode or all modes.
    
    Args:
        mode: Optional mode to get stats for. If None, returns all modes with summary.
    
    Returns:
        Dictionary containing token statistics
    """
    if mode:
        return _token_tracking.get(mode, {})
    else:
        # Return all modes with a summary
        result = _token_tracking.copy()
        
        # Add summary statistics
        summary = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_calls': 0
        }
        
        modes_summary = {}
        for mode_name, data in _token_tracking.items():
            summary['total_input_tokens'] += data['total_input_tokens']
            summary['total_output_tokens'] += data['total_output_tokens'] 
            summary['total_tokens'] += data['total_tokens']
            summary['total_calls'] += data['total_calls']
            
            # Create per-mode summary
            modes_summary[mode_name] = {
                'input_tokens': data['total_input_tokens'],
                'output_tokens': data['total_output_tokens'],
                'total_tokens': data['total_tokens'],
                'calls': data['total_calls']
            }
        
        result['_summary'] = summary
        result['_modes_summary'] = modes_summary
        
        return result
    

def reset_token_tracking(mode: Optional[str] = None):
    """
    Reset token tracking data.
    
    Args:
        mode: Optional mode to reset. If None, resets all tracking data.
    """
    global _token_tracking
    
    if mode:
        if mode in _token_tracking:
            del _token_tracking[mode]
            print(f"Reset tracking data for mode: {mode}")
        else:
            print(f"No tracking data found for mode: {mode}")
    else:
        _token_tracking = {}
        print("Reset all tracking data")


def get_embeddings_stats(mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Get embeddings statistics for a specific mode or all modes.
    
    Args:
        mode: Optional mode to get stats for. If None, returns all modes with summary.
    
    Returns:
        Dictionary containing embeddings statistics
    """
    if mode:
        return _token_tracking_embeddings.get(mode, {})
    else:
        # Return all modes with a summary
        result = _token_tracking_embeddings.copy()
        
        # Add summary statistics
        summary = {
            'total_embeddings_count': 0,
            'total_estimated_tokens': 0,
            'total_calls': 0
        }
        
        modes_summary = {}
        for mode_name, data in _token_tracking_embeddings.items():
            summary['total_embeddings_count'] += data['total_embeddings_count']
            summary['total_estimated_tokens'] += data['total_estimated_tokens']
            summary['total_calls'] += data['total_calls']
            
            # Create per-mode summary
            modes_summary[mode_name] = {
                'embeddings_count': data['total_embeddings_count'],
                'estimated_tokens': data['total_estimated_tokens'],
                'calls': data['total_calls']
            }
        
        result['_summary'] = summary
        result['_modes_summary'] = modes_summary
        
        return result
    

def reset_embeddings_tracking(mode: Optional[str] = None):
    """
    Reset embeddings tracking data.
    
    Args:
        mode: Optional mode to reset. If None, resets all tracking data.
    """
    global _token_tracking_embeddings
    
    if mode:
        if mode in _token_tracking_embeddings:
            del _token_tracking_embeddings[mode]
            print(f"Reset embeddings tracking data for mode: {mode}")
        else:
            print(f"No embeddings tracking data found for mode: {mode}")
    else:
        _token_tracking_embeddings = {}
        print("Reset all embeddings tracking data")


def get_crew_metrics() -> List[Dict[str, Any]]:
    """
    Get all stored crew metrics.
    
    Returns:
        List[Dict[str, Any]]: List of all metrics collected from crew executions
    """
    global _crew_metrics
    return _crew_metrics.copy()  # Return a copy to prevent external modification


def reset_crew_metrics() -> None:
    """
    Reset all stored crew metrics and execution counter.
    """
    global _crew_counter, _crew_metrics
    _crew_counter = 0
    _crew_metrics = []
    print("🔄 Crew metrics and counter have been reset.")