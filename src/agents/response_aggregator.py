"""
Response Aggregator component.
Aggregates responses from multiple agents into a coherent final response.
"""
from typing import Dict, Any, List, Optional
from ..core.tool_enabled_ai import AI
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate


class ResponseAggregator:
    """
    Component responsible for aggregating responses from multiple agents
    into a coherent final response for the user.
    """
    
    def __init__(self, 
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 model: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the ResponseAggregator.
        
        Args:
            unified_config: UnifiedConfig instance
            logger: Logger instance
            model: The model to use for aggregation (or None for default)
            prompt_template: PromptTemplate service (or None to create new)
        """
        self._config = unified_config or UnifiedConfig.get_instance()
        self._logger = logger or LoggerFactory.create(name="response_aggregator")
        
        # Get configuration
        self._agent_config = self._config.get_agent_config("response_aggregator")
        
        # Set up prompt template
        self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
        
        # Set up AI instance for aggregation
        self._ai = AI(
            model=model or self._agent_config.get("default_model"),
            system_prompt=self._get_system_prompt(),
            logger=self._logger
        )
    
    def aggregate_responses(self, 
                           agent_responses: List[Dict[str, Any]], 
                           original_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents.
        
        Args:
            agent_responses: List of agent responses
            original_request: The original request
            
        Returns:
            Aggregated response
            
        Raises:
            AIAgentError: If aggregation fails
        """
        if not agent_responses:
            return {
                "content": "No agents were able to process your request.",
                "agent_id": "response_aggregator",
                "status": "error"
            }
        
        # If only one response, just return it with minimal processing
        if len(agent_responses) == 1:
            response = agent_responses[0]["response"]
            return self._enrich_response(response, [agent_responses[0]["agent_id"]])
        
        # Multiple responses need aggregation
        try:
            # Format responses for the prompt
            responses_text = self._format_responses_for_prompt(agent_responses)
            
            # Prepare template variables
            variables = {
                "prompt": original_request.get("prompt", ""),
                "responses_text": responses_text
            }
            
            # Use template to generate prompt
            try:
                prompt, usage_id = self._prompt_template.render_prompt(
                    template_id="aggregate_responses",
                    variables=variables
                )
            except ValueError:
                # Fallback if template not found
                self._logger.warning("Template 'aggregate_responses' not found, using fallback")
                prompt = self._create_aggregation_prompt(responses_text, original_request)
                usage_id = None
            
            # Get aggregated response
            self._logger.info(f"Aggregating {len(agent_responses)} responses...")
            content = self._ai.request(prompt)
            
            # Record metrics if we used a template
            if usage_id:
                metrics = {
                    "response_length": len(content),
                    "model": self._ai.get_model_info().get("model_id", "unknown"),
                    "num_responses": len(agent_responses)
                }
                self._prompt_template.record_prompt_performance(usage_id, metrics)
            
            # Create final response
            contributing_agents = [resp.get("agent_id") for resp in agent_responses]
            return {
                "content": content,
                "agent_id": "response_aggregator",
                "contributing_agents": contributing_agents,
                "status": "success"
            }
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to aggregate responses: {str(e)}", agent_id="response_aggregator"),
                self._logger
            )
            self._logger.error(f"Aggregation error: {error_response['message']}")
            
            # Fallback to highest confidence response
            self._logger.info("Falling back to highest confidence response")
            sorted_responses = sorted(agent_responses, key=lambda r: r.get("confidence", 0), reverse=True)
            best_response = sorted_responses[0]["response"]
            return self._enrich_response(best_response, [sorted_responses[0]["agent_id"]])
    
    def _format_responses_for_prompt(self, agent_responses: List[Dict[str, Any]]) -> str:
        """
        Format agent responses for inclusion in the prompt.
        
        Args:
            agent_responses: List of agent responses
            
        Returns:
            Formatted string of responses
        """
        formatted = ""
        for i, resp in enumerate(agent_responses, 1):
            agent_id = resp.get("agent_id", "unknown")
            confidence = resp.get("confidence", 0.0)
            content = resp.get("response", {}).get("content", "No content")
            status = resp.get("status", "unknown")
            
            formatted += f"--- Response {i} (Agent: {agent_id}, Confidence: {confidence:.2f}, Status: {status}) ---\n"
            formatted += f"{content}\n\n"
        
        return formatted
    
    def _create_aggregation_prompt(self, responses_text: str, original_request: Dict[str, Any]) -> str:
        """
        Create a prompt for aggregating responses.
        
        Args:
            responses_text: Formatted responses text
            original_request: Original user request
            
        Returns:
            Aggregation prompt
        """
        return f"""
Original user request: {original_request.get('prompt', '')}

Multiple agents have provided responses to this request:

{responses_text}

Create a unified, coherent response that combines the most relevant information from each agent.
Focus on addressing the original request completely and clearly.
Eliminate redundancies and resolve any contradictions between the responses.
Make the response flow naturally as if it came from a single source.
"""
    
    def _enrich_response(self, response: Dict[str, Any], contributing_agents: List[str]) -> Dict[str, Any]:
        """
        Enrich a single response with metadata.
        
        Args:
            response: Original response
            contributing_agents: List of contributing agent IDs
            
        Returns:
            Enriched response
        """
        return {
            "content": response.get("content", ""),
            "agent_id": "response_aggregator",
            "contributing_agents": contributing_agents,
            "status": response.get("status", "success"),
            "metadata": response.get("metadata", {})
        }
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the AI model.
        Uses the template system if available.
        
        Returns:
            System prompt string
        """
        try:
            # Try to use template
            prompt, _ = self._prompt_template.render_prompt(
                template_id="response_aggregator"
            )
            return prompt
        except ValueError:
            # Fallback to hardcoded prompt
            self._logger.warning("Response Aggregator system prompt template not found, using fallback")
            return """You are a Response Aggregator responsible for combining multiple agent responses.
Your task is to create unified, coherent responses that address the user's original request.
Focus on clarity, relevance, and completeness while eliminating redundancies and resolving contradictions.
The final response should flow naturally as if it came from a single source.
""" 