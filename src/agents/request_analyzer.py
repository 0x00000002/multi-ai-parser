"""
Request Analyzer component.
Analyzes user requests to determine appropriate agents and tools.
"""
from typing import Dict, Any, List, Tuple, Optional
import json
import re
from ..core.tool_enabled_ai import AI
from ..config.unified_config import UnifiedConfig
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate


class RequestAnalyzer:
    """
    Component responsible for analyzing user requests and determining
    which agents and tools should handle them.
    """
    
    def __init__(self, 
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 model: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the RequestAnalyzer.
        
        Args:
            unified_config: UnifiedConfig instance
            logger: Logger instance
            model: The model to use for analysis (or None for default)
            prompt_template: PromptTemplate service (or None to create new)
        """
        self._config = unified_config or UnifiedConfig.get_instance()
        self._logger = logger or LoggerFactory.create(name="request_analyzer")
        
        # Get configuration
        self._agent_config = self._config.get_agent_config("request_analyzer")
        self._confidence_threshold = self._agent_config.get("confidence_threshold", 0.6)
        
        # Set up prompt template
        self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
        
        # Set up AI instance for analysis
        self._ai = AI(
            model=model or self._agent_config.get("default_model"),
            system_prompt=self._get_system_prompt(),
            logger=self._logger
        )
    
    def analyze_request(self, 
                       request: Dict[str, Any], 
                       available_agents: List[str], 
                       agent_descriptions: Dict[str, str]) -> List[Tuple[str, float]]:
        """
        Analyze a request to determine appropriate agents.
        
        Args:
            request: The request object
            available_agents: List of available agent IDs
            agent_descriptions: Map of agent IDs to descriptions
            
        Returns:
            List of (agent_id, confidence) tuples sorted by confidence (descending)
            
        Raises:
            AIAgentError: If the analysis fails
        """
        try:
            # Format agent list for template
            agent_list = self._format_agent_list(available_agents, agent_descriptions)
            
            # Prepare template variables
            variables = {
                "prompt": request.get("prompt", ""),
                "agent_list": agent_list,
                "confidence_threshold": self._confidence_threshold
            }
            
            # Use template to generate prompt
            try:
                prompt, usage_id = self._prompt_template.render_prompt(
                    template_id="analyze_request",
                    variables=variables
                )
            except ValueError:
                # Fallback if template not found
                self._logger.warning("Template 'analyze_request' not found, using fallback")
                prompt = self._create_analysis_prompt(request, available_agents, agent_descriptions)
                usage_id = None
            
            # Get response from AI
            self._logger.info(f"Analyzing request: {request.get('prompt', '')[:50]}...")
            response = self._ai.request(prompt)
            
            # Record metrics if we used a template
            if usage_id:
                metrics = {
                    "response_length": len(response),
                    "model": self._ai.get_model_info().get("model_id", "unknown")
                }
                self._prompt_template.record_prompt_performance(usage_id, metrics)
            
            # Parse response
            agents = self._parse_agent_assignments(response)
            
            # Filter by confidence threshold
            filtered_agents = [(agent_id, confidence) 
                              for agent_id, confidence in agents 
                              if confidence >= self._confidence_threshold]
            
            # Sort by confidence (descending)
            sorted_agents = sorted(filtered_agents, key=lambda x: x[1], reverse=True)
            
            self._logger.info(f"Analysis result: {sorted_agents}")
            return sorted_agents
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to analyze request: {str(e)}", agent_id="request_analyzer"),
                self._logger
            )
            self._logger.error(f"Analysis error: {error_response['message']}")
            return []
    
    def analyze_tools(self, 
                     request: Dict[str, Any], 
                     available_tools: List[str], 
                     tool_descriptions: Dict[str, str]) -> List[str]:
        """
        Analyze a request to determine appropriate tools.
        
        Args:
            request: The request object
            available_tools: List of available tool IDs
            tool_descriptions: Map of tool IDs to descriptions
            
        Returns:
            List of tool IDs that should be used
            
        Raises:
            AIAgentError: If the analysis fails
        """
        try:
            # Format tool list for template
            tool_list = self._format_tool_list(available_tools, tool_descriptions)
            
            # Prepare template variables
            variables = {
                "prompt": request.get("prompt", ""),
                "tool_list": tool_list
            }
            
            # Use template to generate prompt
            try:
                prompt, usage_id = self._prompt_template.render_prompt(
                    template_id="analyze_tools",
                    variables=variables
                )
            except ValueError:
                # Fallback if template not found
                self._logger.warning("Template 'analyze_tools' not found, using fallback")
                prompt = self._create_tool_analysis_prompt(request, available_tools, tool_descriptions)
                usage_id = None
            
            # Get response from AI
            self._logger.info(f"Analyzing tools for request: {request.get('prompt', '')[:50]}...")
            response = self._ai.request(prompt)
            
            # Record metrics if we used a template
            if usage_id:
                metrics = {
                    "response_length": len(response),
                    "model": self._ai.get_model_info().get("model_id", "unknown")
                }
                self._prompt_template.record_prompt_performance(usage_id, metrics)
            
            # Parse response
            tools = self._parse_tool_assignments(response)
            
            self._logger.info(f"Tool analysis result: {tools}")
            return tools
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Failed to analyze tools: {str(e)}", agent_id="request_analyzer"),
                self._logger
            )
            self._logger.error(f"Tool analysis error: {error_response['message']}")
            return []
    
    def _format_agent_list(self, available_agents: List[str], agent_descriptions: Dict[str, str]) -> str:
        """
        Format the agent list for inclusion in a prompt.
        
        Args:
            available_agents: List of available agent IDs
            agent_descriptions: Map of agent IDs to descriptions
            
        Returns:
            Formatted string of agents
        """
        agent_list = ""
        for agent_id in available_agents:
            description = agent_descriptions.get(agent_id, f"Agent type: {agent_id}")
            agent_list += f"- {agent_id}: {description}\n"
        return agent_list
    
    def _format_tool_list(self, available_tools: List[str], tool_descriptions: Dict[str, str]) -> str:
        """
        Format the tool list for inclusion in a prompt.
        
        Args:
            available_tools: List of available tool IDs
            tool_descriptions: Map of tool IDs to descriptions
            
        Returns:
            Formatted string of tools
        """
        tool_list = ""
        for tool_id in available_tools:
            description = tool_descriptions.get(tool_id, f"Tool: {tool_id}")
            tool_list += f"- {tool_id}: {description}\n"
        return tool_list
    
    def _create_analysis_prompt(self, 
                               request: Dict[str, Any], 
                               available_agents: List[str], 
                               agent_descriptions: Dict[str, str]) -> str:
        """
        Create a prompt for analyzing the request.
        
        Args:
            request: The request object
            available_agents: List of available agent IDs
            agent_descriptions: Map of agent IDs to descriptions
            
        Returns:
            Prompt for analysis
        """
        # Format agent list for prompt
        agent_list = self._format_agent_list(available_agents, agent_descriptions)
        
        prompt = f"""Analyze this user request and determine which specialized agents should handle it:
        
Request: {request.get('prompt', '')}

Available agents:
{agent_list}

Return a JSON list of [agent_id, confidence] pairs, where confidence is 0.0-1.0.
Only include agents with confidence > {self._confidence_threshold}. If no agents are appropriate, return [].
"""
        
        return prompt
    
    def _create_tool_analysis_prompt(self, 
                                    request: Dict[str, Any], 
                                    available_tools: List[str], 
                                    tool_descriptions: Dict[str, str]) -> str:
        """
        Create a prompt for analyzing which tools should be used.
        
        Args:
            request: The request object
            available_tools: List of available tool IDs
            tool_descriptions: Map of tool IDs to descriptions
            
        Returns:
            Prompt for tool analysis
        """
        # Format tool list for prompt
        tool_list = self._format_tool_list(available_tools, tool_descriptions)
        
        prompt = f"""Analyze this user request and determine which tools would be helpful:
        
Request: {request.get('prompt', '')}

Available tools:
{tool_list}

Return a JSON list of tool IDs that would be helpful for this request.
If no tools are needed, return [].
"""
        
        return prompt
    
    def _parse_agent_assignments(self, response: str) -> List[Tuple[str, float]]:
        """
        Parse the agent assignment response from the AI.
        
        Args:
            response: AI response string
            
        Returns:
            List of (agent_id, confidence) tuples
        """
        try:
            # Try to parse as JSON list of [agent_id, confidence] pairs
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [(str(item[0]), float(item[1])) for item in parsed 
                       if len(item) == 2 and isinstance(item[1], (int, float))]
            else:
                self._logger.warning(f"Invalid response format, expected list: {response}")
                return []
                
        except json.JSONDecodeError:
            self._logger.warning(f"Failed to parse JSON response: {response}")
            
            # Fallback: Try to extract agent assignments using regex
            pattern = r'["\']([\w_]+)["\'],\s*(0\.\d+)'
            matches = re.findall(pattern, response)
            
            if matches:
                return [(match[0], float(match[1])) for match in matches]
            else:
                return []
        except Exception as e:
            self._logger.error(f"Error parsing agent assignments: {str(e)}")
            return []
    
    def _parse_tool_assignments(self, response: str) -> List[str]:
        """
        Parse the tool assignment response from the AI.
        
        Args:
            response: AI response string
            
        Returns:
            List of tool IDs
        """
        try:
            # Try to parse as JSON list of tool_ids
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if isinstance(item, (str, int))]
            else:
                self._logger.warning(f"Invalid response format, expected list: {response}")
                return []
                
        except json.JSONDecodeError:
            self._logger.warning(f"Failed to parse JSON response: {response}")
            
            # Fallback: Try to extract tool IDs using regex
            pattern = r'["\']([\w_]+)["\']'
            matches = re.findall(pattern, response)
            
            if matches:
                return matches
            else:
                return []
        except Exception as e:
            self._logger.error(f"Error parsing tool assignments: {str(e)}")
            return []
    
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
                template_id="request_analyzer"
            )
            return prompt
        except ValueError:
            # Fallback to hardcoded prompt
            self._logger.warning("Request Analyzer system prompt template not found, using fallback")
            return """You are a Request Analyzer responsible for analyzing user requests.
Your task is to determine which specialized agents or tools should handle user requests.
Analyze the main intent and required capabilities to make accurate assignments.
Only recommend agents or tools that are highly relevant to the user's request.
""" 