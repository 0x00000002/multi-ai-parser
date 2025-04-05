import time
import functools
from typing import Callable, Any, Optional
from .request_metrics import RequestMetricsService

metrics_service = RequestMetricsService()

def track_tool_execution(tool_id: str):
    """
    Decorator to track tool execution metrics.
    
    Args:
        tool_id: ID of the tool being executed
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_id = kwargs.get('request_id')
            if not request_id:
                # Try to extract from first argument which might be a request object
                if args and hasattr(args[0], 'request_id'):
                    request_id = args[0].request_id
            
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)
                
                if request_id:
                    metrics_service.track_tool_usage(
                        request_id=request_id,
                        tool_id=tool_id,
                        duration_ms=duration_ms,
                        success=success,
                        metadata={"error": error} if error else None
                    )
        
        return wrapper
    
    return decorator

def track_agent_execution(agent_id: str):
    """
    Decorator to track agent execution metrics.
    
    Args:
        agent_id: ID of the agent being executed
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            confidence = kwargs.get('confidence', 1.0)
            
            # Extract request_id from request object
            request_id = None
            if request:
                if isinstance(request, dict):
                    request_id = request.get('request_id')
                elif hasattr(request, 'request_id'):
                    request_id = request.request_id
            
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)
                
                if request_id:
                    metrics_service.track_agent_usage(
                        request_id=request_id,
                        agent_id=agent_id,
                        confidence=confidence,
                        duration_ms=duration_ms,
                        success=success,
                        metadata={"error": error} if error else None
                    )
        
        return wrapper
    
    return decorator