"""
Orchestrator for the multi-agent architecture.
Coordinates the workflow between specialized agents.
"""
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface
from ..config.unified_config import UnifiedConfig
from .tool_finder_agent import ToolFinderAgent
from .request_analyzer import RequestAnalyzer
from .response_aggregator import ResponseAggregator
from ..tools.tool_registry import ToolRegistry
from .interfaces import AgentResponse, AgentResponseStatus
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate
from ..core.model_selector import ModelSelector, UseCase
from ..core.tool_enabled_ai import AI
from ..metrics.request_metrics import RequestMetricsService
import time


class Orchestrator(BaseAgent):
    """
    Orchestrator agent responsible for coordinating the workflow between specialized agents.
    Serves as the entry point for all user interactions in the multi-agent system.
    """
    
    def __init__(self, 
                 agent_factory=None,
                 tool_finder_agent=None,
                 request_analyzer=None,
                 response_aggregator=None,
                 unified_config=None,
                 logger=None,
                 prompt_template=None,
                 model_selector=None,
                 **kwargs):
        """
        Initialize the orchestrator.
        
        Args:
            agent_factory: Factory for creating agent instances
            tool_finder_agent: Tool finder agent instance
            request_analyzer: Request analyzer component
            response_aggregator: Response aggregator component
            unified_config: UnifiedConfig instance
            logger: Logger instance
            prompt_template: PromptTemplate service for generating prompts
            model_selector: ModelSelector for intelligent model selection
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id="orchestrator", unified_config=unified_config, logger=logger, **kwargs)
        
        # Core components
        self.agent_factory = agent_factory
        self.tool_finder_agent = tool_finder_agent
        self.request_analyzer = request_analyzer
        self.response_aggregator = response_aggregator
        
        # Set up prompt template
        self._prompt_template = prompt_template or PromptTemplate(logger=self.logger)
        
        # Set up model selector
        self.model_selector = model_selector or ModelSelector()
        
        # Get configuration
        self.max_parallel_agents = self.agent_config.get("max_parallel_agents", 3)
        
        # Validate dependencies
        if self.agent_factory is None:
            self.logger.warning("No agent factory provided, will be unable to route requests")
            
        # Create missing components if needed
        if self.request_analyzer is None:
            self.logger.info("Creating default RequestAnalyzer")
            self.request_analyzer = RequestAnalyzer(
                unified_config=self.config,
                logger=self.logger,
                prompt_template=self._prompt_template
            )
            
        if self.response_aggregator is None:
            self.logger.info("Creating default ResponseAggregator")
            self.response_aggregator = ResponseAggregator(
                unified_config=self.config,
                logger=self.logger,
                prompt_template=self._prompt_template
            )
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request by orchestrating specialized agents.
        
        Args:
            request: The request object containing prompt and metadata
            
        Returns:
            Response object with content and metadata
        """
        # Import request metrics service
        metrics_service = RequestMetricsService()
        
        # Generate request ID if not present
        if "request_id" not in request:
            import uuid
            request["request_id"] = str(uuid.uuid4())
            
        # Start tracking the request
        metrics_service.start_request_tracking(
            request_id=request["request_id"],
            prompt=request.get("prompt", ""),
            metadata={"user_id": request.get("user_id")}
        )
        
        try:
            # Ensure request is properly formatted
            if not isinstance(request, dict):
                self.logger.warning(f"Request is not a dictionary: {type(request)}")
                request = {"prompt": str(request), "request_id": request["request_id"]}
                
            # Make sure prompt exists in request
            if "prompt" not in request:
                self.logger.warning("No prompt in request, using empty string")
                request["prompt"] = ""
                
            self.logger.info(f"Processing request: {request.get('prompt', '')[:50]}...")
            
            # Step 0: Determine the use case for this request
            use_case = self._determine_use_case(request)
            if use_case:
                self.logger.info(f"Determined use case: {use_case.name}")
                # Add use case to request for agents
                request["use_case"] = use_case.name
                
                # Select appropriate model based on use case
                try:
                    # Get system prompt for this use case
                    system_prompt = self.model_selector.get_system_prompt(use_case)
                    request["system_prompt"] = system_prompt
                    
                    # Get the best model for this use case
                    model = self.model_selector.select_model(use_case)
                    if model:
                        self.logger.info(f"Selected model for request: {model.value}")
                        request["model"] = model.value
                        
                        # Track model selection
                        metrics_service.track_model_usage(
                            request_id=request["request_id"],
                            model_id=model.value,
                            metadata={"use_case": use_case.name}
                        )
                        
                        # If we have an AI instance and no specialized agents will handle this,
                        # update the model directly on our AI instance
                        if not self.ai_instance:
                            self.logger.info("Creating new AI instance with selected model")
                            self.ai_instance = AI(
                                model=model.value,
                                system_prompt=system_prompt,
                                logger=self.logger
                            )
                except Exception as e:
                    self.logger.warning(f"Error selecting model for use case {use_case}: {str(e)}")
            
            # Step 1: Find relevant tools for the request
            start_time = time.time()
            try:
                relevant_tools = self._find_relevant_tools(request)
                self.logger.info(f"Found {len(relevant_tools)} relevant tools")
                
                # Track tool finder agent usage
                if self.tool_finder_agent:
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id="tool_finder",
                        duration_ms=int((time.time() - start_time) * 1000),
                        success=True
                    )
            except Exception as e:
                self.logger.error(f"Error finding relevant tools: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                relevant_tools = []
                
                # Track tool finder failure
                if self.tool_finder_agent:
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id="tool_finder",
                        duration_ms=int((time.time() - start_time) * 1000),
                        success=False,
                        metadata={"error": str(e)}
                    )
            
            # Step 2: Analyze request to determine appropriate agents
            start_time = time.time()
            try:
                agent_assignments = self._analyze_request(request)
                self.logger.info(f"Analyzed request, found {len(agent_assignments)} agent assignments")
                
                # Track request analyzer usage
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id="request_analyzer",
                    duration_ms=int((time.time() - start_time) * 1000),
                    success=True,
                    metadata={"num_agents_found": len(agent_assignments)}
                )
            except Exception as e:
                self.logger.error(f"Error analyzing request: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                agent_assignments = []
                
                # Track request analyzer failure
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id="request_analyzer",
                    duration_ms=int((time.time() - start_time) * 1000),
                    success=False,
                    metadata={"error": str(e)}
                )
            
            # If no agents identified, handle directly
            if not agent_assignments:
                self.logger.info("No specialized agents identified for request, handling directly")
                
                # Track direct handling by orchestrator
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id=self.agent_id,
                    confidence=1.0,
                    metadata={"direct_handling": True}
                )
                
                result = super().process_request(request)
                
                # End request tracking
                metrics_service.end_request_tracking(
                    request_id=request["request_id"],
                    success=True if "error" not in result else False,
                    error=result.get("error")
                )
                
                return result
            
            # Step 3: Process with identified agents
            try:
                agent_responses = self._process_with_agents(request, agent_assignments, relevant_tools, use_case)
                self.logger.info(f"Processed with {len(agent_responses)} agents")
            except Exception as e:
                self.logger.error(f"Error processing with agents: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                # End request tracking with error
                metrics_service.end_request_tracking(
                    request_id=request["request_id"],
                    success=False,
                    error=f"Error processing with agents: {str(e)}"
                )
                
                return {
                    "content": f"An error occurred while processing your request with specialized agents: {str(e)}",
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": str(e)
                }
            
            # Step 4: Aggregate responses
            start_time = time.time()
            try:
                aggregated_response = self._aggregate_responses(agent_responses, request)
                self.logger.info("Successfully aggregated responses")
                
                # Track response aggregator usage
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id="response_aggregator",
                    duration_ms=int((time.time() - start_time) * 1000),
                    success=True
                )
                
                # End request tracking
                metrics_service.end_request_tracking(
                    request_id=request["request_id"],
                    success=True if "error" not in aggregated_response else False,
                    error=aggregated_response.get("error")
                )
                
                # Add metrics data to the response
                if "metadata" not in aggregated_response:
                    aggregated_response["metadata"] = {}
                
                aggregated_response["metadata"]["request_id"] = request["request_id"]
                aggregated_response["metadata"]["agents_used"] = [a["agent_id"] for a in agent_responses]
                aggregated_response["metadata"]["tools_used"] = relevant_tools
                
                return aggregated_response
            except Exception as e:
                self.logger.error(f"Error aggregating responses: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                # Track response aggregator failure
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id="response_aggregator",
                    duration_ms=int((time.time() - start_time) * 1000),
                    success=False,
                    metadata={"error": str(e)}
                )
                
                # End request tracking with error
                metrics_service.end_request_tracking(
                    request_id=request["request_id"],
                    success=False,
                    error=f"Error aggregating responses: {str(e)}"
                )
                
                return {
                    "content": f"An error occurred while aggregating responses: {str(e)}",
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": str(e)
                }
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Error orchestrating request: {str(e)}", agent_id="orchestrator"),
                self.logger
            )
            self.logger.error(f"Orchestration error: {error_response['message']}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # End request tracking with error
            try:
                metrics_service.end_request_tracking(
                    request_id=request.get("request_id", "unknown"),
                    success=False,
                    error=f"Orchestration error: {str(e)}"
                )
            except:
                pass
            
            return {
                "content": f"An error occurred while processing your request: {str(e)}",
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e)
            }
    
    def _determine_use_case(self, request: Dict[str, Any]) -> Optional[UseCase]:
        """
        Determine the use case for this request using heuristics.
        
        Args:
            request: The request object
            
        Returns:
            UseCase enum or None if unable to determine
        """
        prompt = request.get("prompt", "").lower()
        
        # Check for explicitly requested use case
        if "use_case" in request and isinstance(request["use_case"], str):
            try:
                return UseCase.from_string(request["use_case"])
            except ValueError:
                self.logger.warning(f"Invalid use case specified: {request['use_case']}")
        
        # Check for code-related keywords
        code_keywords = ["code", "function", "programming", "algorithm", "implementation", 
                         "debug", "refactor", "javascript", "python", "java", "typescript",
                         "c++", "rust", "go", "coding"]
        
        # Check for Solidity-specific keywords
        solidity_keywords = ["solidity", "smart contract", "blockchain", "ethereum", "web3", 
                             "token", "defi", "nft", "erc20", "erc721", "gas optimization"]
        
        # Check for translation keywords
        translation_keywords = ["translate", "translation", "convert to", "change to language",
                               "english to", "spanish to", "french to", "german to", "chinese to",
                               "japanese to", "korean to", "russian to", "arabic to"]
        
        # Check for summarization keywords
        summarization_keywords = ["summarize", "summary", "tldr", "summary of", "key points",
                                 "shorten this", "brief overview", "main ideas"]
        
        # Check for data analysis keywords
        data_analysis_keywords = ["analyze data", "data analysis", "statistics", "correlation",
                                 "insights from", "trends in", "data interpretation", "chart", 
                                 "graph", "visualization"]
        
        # Check for web analysis keywords
        web_analysis_keywords = ["analyze website", "webpage analysis", "site review",
                                "web content", "html analysis", "css review", "web performance"]
        
        # Check for content generation keywords
        content_generation_keywords = ["create content", "generate text", "write a", "creative",
                                      "story", "blog post", "article", "content creation"]
        
        # Check for image generation keywords (if supported)
        image_generation_keywords = ["generate image", "create image", "draw", "picture of",
                                    "illustration", "visualization of", "image of"]
        
        # Detect use case based on keywords
        if any(keyword in prompt for keyword in solidity_keywords):
            return UseCase.SOLIDITY_CODING
        elif any(keyword in prompt for keyword in code_keywords):
            return UseCase.CODING
        elif any(keyword in prompt for keyword in translation_keywords):
            return UseCase.TRANSLATION
        elif any(keyword in prompt for keyword in summarization_keywords):
            return UseCase.SUMMARIZATION
        elif any(keyword in prompt for keyword in data_analysis_keywords):
            return UseCase.DATA_ANALYSIS
        elif any(keyword in prompt for keyword in web_analysis_keywords):
            return UseCase.WEB_ANALYSIS
        elif any(keyword in prompt for keyword in content_generation_keywords):
            return UseCase.CONTENT_GENERATION
        elif any(keyword in prompt for keyword in image_generation_keywords):
            return UseCase.IMAGE_GENERATION
            
        # Default to CHAT if no specific use case detected
        return UseCase.CHAT
    
    def _find_relevant_tools(self, request: Dict[str, Any]) -> List[str]:
        """
        Find relevant tools for the request using the tool finder agent.
        
        Args:
            request: The request object
            
        Returns:
            List of relevant tool IDs
        """
        if not self.tool_finder_agent:
            self.logger.warning("No tool finder agent available")
            return []
            
        try:
            self.logger.info("Finding relevant tools...")
            
            # Ensure request is properly formatted
            if not isinstance(request, dict):
                self.logger.warning(f"Request is not a dictionary: {type(request)}")
                request = {"prompt": str(request)}
                
            # Make sure prompt exists in request
            if "prompt" not in request:
                self.logger.warning("No prompt in request, using empty string")
                request["prompt"] = ""
                
            response = self.tool_finder_agent.process_request(request)
            
            # Handle different response formats
            if isinstance(response, dict) and "selected_tools" in response:
                tools = response["selected_tools"]
                self.logger.info(f"Found {len(tools)} tools from dict response")
                return tools
            elif hasattr(response, "selected_tools"):
                tools = response.selected_tools
                self.logger.info(f"Found {len(tools)} tools from object response")
                return tools
            elif hasattr(response, "content") and isinstance(response.content, list):
                self.logger.info(f"Using content as tools list: {response.content}")
                return response.content
            else:
                self.logger.warning(f"Unexpected response format from tool finder agent: {response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error finding relevant tools: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _analyze_request(self, request: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Analyze the request to determine appropriate agents.
        
        Args:
            request: The request object
            
        Returns:
            List of (agent_id, confidence) tuples
        """
        if not self.agent_factory:
            self.logger.warning("No agent factory available, cannot analyze request")
            return []
        
        # Get list of available agents
        available_agents = self.agent_factory.registry.get_all_agents() if self.agent_factory else []
        
        # Get agent descriptions from the config manager
        agent_descriptions = self.config.get_agent_descriptions() if self.config else {}
        
        # Use the RequestAnalyzer to analyze the request
        return self.request_analyzer.analyze_request(
            request=request,
            available_agents=available_agents,
            agent_descriptions=agent_descriptions
        )
    
    def _process_with_agents(self, 
                            request: Dict[str, Any], 
                            agent_assignments: List[Tuple[str, float]],
                            relevant_tools: List[str],
                            use_case: Optional[UseCase] = None) -> List[Dict[str, Any]]:
        """
        Process the request with assigned agents.
        
        Args:
            request: The request object
            agent_assignments: List of (agent_id, confidence) tuples
            relevant_tools: List of relevant tool IDs
            use_case: Optional UseCase identified for the request
            
        Returns:
            List of agent responses
        """
        # Import metrics service
        from ..metrics.request_metrics import RequestMetricsService
        metrics_service = RequestMetricsService()
        
        agent_responses = []
        
        # Limit number of agents to prevent excessive processing
        agent_assignments = agent_assignments[:self.max_parallel_agents]
        
        # Process with each identified agent
        for agent_id, confidence in agent_assignments:
            agent_start_time = time.time()
            success = False
            error_message = None
            
            try:
                self.logger.info(f"Processing with agent {agent_id} (confidence: {confidence})")
                
                # Create a new request with relevant tools
                enriched_request = self._enrich_request(request, relevant_tools)
                
                # Select appropriate model for this agent and use case
                if use_case:
                    try:
                        # Add system prompt based on use case if possible
                        enriched_request["system_prompt"] = self.model_selector.get_system_prompt(use_case)
                        
                        # Add model selection to the request
                        model = self.model_selector.select_model(use_case)
                        if model:
                            self.logger.info(f"Selected model for {agent_id}: {model.value}")
                            enriched_request["model"] = model.value
                            
                            # Track model usage for this agent
                            metrics_service.track_model_usage(
                                request_id=request["request_id"],
                                model_id=model.value,
                                metadata={
                                    "use_case": use_case.name,
                                    "agent_id": agent_id
                                }
                            )
                    except Exception as e:
                        self.logger.warning(f"Error selecting model for use case {use_case}: {str(e)}")
                
                # Get agent instance
                if not self.agent_factory:
                    self.logger.error("Agent factory not available, cannot create agent")
                    continue
                    
                agent = self.agent_factory.create(agent_id)
                
                if not agent:
                    self.logger.error(f"Failed to create agent {agent_id}")
                    continue
                
                # Pass request_id to agent if needed
                if "request_id" not in enriched_request and "request_id" in request:
                    enriched_request["request_id"] = request["request_id"]
                
                # Process the request
                try:
                    response = agent.process_request(enriched_request)
                    success = True
                except Exception as e:
                    self.logger.error(f"Error in agent {agent_id} process_request: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    response = {
                        "content": f"Error: {str(e)}",
                        "status": "error"
                    }
                    error_message = str(e)
                
                # Normalize the response format
                try:
                    normalized_response = self._normalize_response(response)
                except Exception as e:
                    self.logger.error(f"Error normalizing response from agent {agent_id}: {str(e)}")
                    normalized_response = {
                        "content": f"Error normalizing response: {str(e)}",
                        "status": "error"
                    }
                    success = False
                    error_message = str(e)
                
                # Add to responses
                agent_responses.append({
                    "agent_id": agent_id,
                    "confidence": confidence,
                    "response": normalized_response,
                    "status": normalized_response.get("status", "success")
                })
                
                # Track successful agent usage
                if success:
                    metrics_service.track_agent_usage(
                        request_id=request["request_id"],
                        agent_id=agent_id,
                        confidence=confidence,
                        duration_ms=int((time.time() - agent_start_time) * 1000),
                        success=True
                    )
                
            except Exception as e:
                error_response = ErrorHandler.handle_error(
                    AIAgentError(f"Error with agent {agent_id}: {str(e)}", agent_id=agent_id),
                    self.logger
                )
                self.logger.error(f"Agent processing error: {error_response['message']}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                agent_responses.append({
                    "agent_id": agent_id,
                    "confidence": confidence,
                    "response": {"content": f"Error: {str(e)}", "status": "error"},
                    "status": "error",
                    "error": str(e)
                })
                
                error_message = str(e)
                
            # Track failed agent usage if not already tracked
            if not success:
                metrics_service.track_agent_usage(
                    request_id=request["request_id"],
                    agent_id=agent_id,
                    confidence=confidence,
                    duration_ms=int((time.time() - agent_start_time) * 1000),
                    success=False,
                    metadata={"error": error_message}
                )
        
        return agent_responses
    
    def _enrich_request(self, request: Dict[str, Any], relevant_tools: List[str]) -> Dict[str, Any]:
        """
        Enrich the request with additional context.
        
        Args:
            request: The original request
            relevant_tools: List of relevant tool IDs
            
        Returns:
            Enriched request
        """
        # Create a copy to avoid modifying the original
        enriched = dict(request)
        
        # Add relevant tools
        if relevant_tools:
            enriched["relevant_tools"] = relevant_tools
            
        # Add orchestrator context
        if "context" not in enriched:
            enriched["context"] = {}
            
        enriched["context"]["orchestrator_id"] = self.agent_id
        
        return enriched
    
    def _normalize_response(self, response: Any) -> Dict[str, Any]:
        """
        Normalize agent responses to a consistent format.
        
        Args:
            response: Agent response (various formats)
            
        Returns:
            Normalized response dictionary
        """
        # Handle AgentResponse objects
        if hasattr(response, "status") and hasattr(response, "content"):
            return {
                "content": response.content,
                "status": response.status.value if hasattr(response.status, "value") else str(response.status),
                "metadata": getattr(response, "metadata", {})
            }
        
        # Handle dictionaries
        if isinstance(response, dict):
            if "content" in response:
                return response
            else:
                return {"content": str(response), "status": "success"}
        
        # Handle strings
        if isinstance(response, str):
            return {"content": response, "status": "success"}
        
        # Handle other types
        return {"content": str(response), "status": "success"}
    
    def _aggregate_responses(self, 
                           agent_responses: List[Dict[str, Any]], 
                           original_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents using the ResponseAggregator.
        
        Args:
            agent_responses: List of agent responses
            original_request: The original request
            
        Returns:
            Aggregated response
        """
        if not agent_responses:
            self.logger.warning("No agent responses to aggregate")
            return {
                "content": "No agents were able to process your request.",
                "agent_id": self.agent_id,
                "status": "error"
            }
        
        # Check if response aggregator is available
        if not self.response_aggregator:
            self.logger.error("Response aggregator not available")
            return {
                "content": "Error: Response aggregator not available",
                "agent_id": self.agent_id,
                "status": "error"
            }
        
        try:
            # Use the ResponseAggregator to aggregate responses
            aggregated_response = self.response_aggregator.aggregate_responses(
                agent_responses=agent_responses,
                original_request=original_request
            )
            
            # Ensure the response has the required fields
            if not isinstance(aggregated_response, dict):
                self.logger.warning(f"Aggregated response is not a dictionary: {type(aggregated_response)}")
                return {
                    "content": str(aggregated_response),
                    "agent_id": self.agent_id,
                    "status": "success"
                }
                
            if "content" not in aggregated_response:
                self.logger.warning("Aggregated response missing 'content' field")
                aggregated_response["content"] = "No content available"
                
            if "agent_id" not in aggregated_response:
                aggregated_response["agent_id"] = self.agent_id
                
            if "status" not in aggregated_response:
                aggregated_response["status"] = "success"
                
            return aggregated_response
            
        except Exception as e:
            self.logger.error(f"Error aggregating responses: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return a fallback response with the first agent's content
            if agent_responses and "response" in agent_responses[0] and "content" in agent_responses[0]["response"]:
                return {
                    "content": agent_responses[0]["response"]["content"],
                    "agent_id": self.agent_id,
                    "status": "partial",
                    "note": "Using first agent response due to aggregation error"
                }
            else:
                return {
                    "content": "An error occurred while aggregating responses from multiple agents.",
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": str(e)
                }
    
    def set_tool_finder_agent(self, tool_finder_agent: ToolFinderAgent) -> None:
        """
        Set the tool finder agent.
        
        Args:
            tool_finder_agent: Tool finder agent instance
        """
        self.tool_finder_agent = tool_finder_agent
        self.logger.info("Tool finder agent set")
    
    def set_request_analyzer(self, request_analyzer: RequestAnalyzer) -> None:
        """
        Set the request analyzer.
        
        Args:
            request_analyzer: Request analyzer instance
        """
        self.request_analyzer = request_analyzer
        self.logger.info("Request analyzer set")
    
    def set_response_aggregator(self, response_aggregator: ResponseAggregator) -> None:
        """
        Set the response aggregator.
        
        Args:
            response_aggregator: Response aggregator instance
        """
        self.response_aggregator = response_aggregator
        self.logger.info("Response aggregator set")
    
    def set_model_selector(self, model_selector: ModelSelector) -> None:
        """
        Set the model selector.
        
        Args:
            model_selector: ModelSelector instance
        """
        self.model_selector = model_selector
        self.logger.info("Model selector set")
    
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
                template_id="orchestrator"
            )
            return prompt
        except ValueError:
            # Fallback to hardcoded prompt
            self.logger.warning("Orchestrator system prompt template not found, using fallback")
            return """You are the Orchestrator agent, responsible for coordinating the multi-agent system.
Your task is to handle user requests that don't require specialized agents,
providing helpful responses based on available tools and your general knowledge.
"""