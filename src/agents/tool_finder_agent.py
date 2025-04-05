"""
Tool Finder Agent.
Identifies relevant tools for user requests.
"""
from typing import List, Dict, Any, Optional
import json
from .base_agent import BaseAgent
from ..tools.tool_registry import ToolRegistry
from .interfaces import AgentResponse, AgentResponseStatus
from ..exceptions import AIAgentError, ErrorHandler
from ..prompts.prompt_template import PromptTemplate


class ToolFinderAgent(BaseAgent):
    """
    Agent responsible for finding relevant tools for user requests.
    """
    
    def __init__(self, 
                 agent_id: str = "tool_finder", 
                 unified_config=None,
                 logger=None,
                 tool_registry=None,
                 prompt_template=None,
                 **kwargs):
        """
        Initialize the tool finder agent.
        
        Args:
            agent_id: ID of the agent
            unified_config: UnifiedConfig instance
            logger: Logger instance
            tool_registry: Tool registry instance
            prompt_template: PromptTemplate service for generating prompts
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, unified_config=unified_config, logger=logger, **kwargs)
        
        # Set up tool registry
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Set up prompt template
        self._prompt_template = prompt_template or PromptTemplate(logger=self.logger)
    
    def process_request(self, request) -> AgentResponse:
        """
        Process a request to find relevant tools.
        
        Args:
            request: Request object
            
        Returns:
            AgentResponse with selected tools
        """
        try:
            # Get tools from registry
            available_tools = self.tool_registry.get_tools()
            
            # Format tools for prompt
            tools_str = self._format_tools_for_prompt(available_tools)
            
            # Prepare template variables
            variables = {
                "tools_str": tools_str,
                "prompt": request.prompt
            }
            
            # Use template to generate prompt
            try:
                prompt, usage_id = self._prompt_template.render_prompt(
                    template_id="find_tools",
                    variables=variables
                )
            except ValueError:
                # Fallback if template not found
                self.logger.warning("Template 'find_tools' not found, using fallback")
                prompt = self._create_tool_finding_prompt(tools_str, request.prompt)
                usage_id = None
            
            # Get response from AI
            response = self.ai.request(prompt)
            
            # Record metrics if we used a template
            if usage_id:
                metrics = {
                    "response_length": len(response),
                    "num_tools": len(available_tools),
                    "model": getattr(self.ai, "model_id", "unknown")
                }
                self._prompt_template.record_prompt_performance(usage_id, metrics)
            
            # Parse the response to get tool names
            tool_names = self._parse_tool_names(response)
            
            # Update usage stats for recommended tools
            for tool_name in tool_names:
                if tool_name in available_tools:
                    self.tool_registry.update_usage_stats(tool_name, True)
            
            return AgentResponse(
                status=AgentResponseStatus.SUCCESS,
                content=f"Found {len(tool_names)} relevant tools",
                selected_tools=tool_names
            )
            
        except Exception as e:
            error_response = ErrorHandler.handle_error(
                AIAgentError(f"Error finding tools: {str(e)}", agent_id=self.agent_id),
                self.logger
            )
            self.logger.error(f"Tool finding error: {error_response['message']}")
            
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                content=f"Error finding tools: {str(e)}",
                selected_tools=[]
            )
    
    def _create_tool_finding_prompt(self, tools_str: str, user_prompt: str) -> str:
        """
        Create a fallback prompt for finding tools if the template is not available.
        
        Args:
            tools_str: Formatted string of available tools
            user_prompt: User's request prompt
            
        Returns:
            Prompt for finding tools
        """
        return f"""Available tools:
{tools_str}

User request: {user_prompt}

Which tools would be most useful for this request? Respond with a JSON array of tool names only.
"""
    
    def _format_tools_for_prompt(self, tools: Dict[str, Any]) -> str:
        """
        Format tools for inclusion in the prompt.
        
        Args:
            tools: Dictionary of tools
            
        Returns:
            Formatted string of tools
        """
        formatted_tools = []
        for name, tool in tools.items():
            description = getattr(tool, 'description', 'No description available')
            formatted_tools.append(f"- {name}: {description}")
        
        return "\n".join(formatted_tools)
    
    def _parse_tool_names(self, response: str) -> List[str]:
        """
        Parse tool names from the AI response.
        
        Args:
            response: AI response string
            
        Returns:
            List of tool names
        """
        try:
            # Try to parse as JSON
            tool_names = json.loads(response)
            
            # Ensure it's a list
            if not isinstance(tool_names, list):
                self.logger.warning(f"Expected JSON array, got {type(tool_names)}")
                return []
                
            # Validate tool names
            valid_tools = []
            available_tools = self.tool_registry.get_tool_names()
            
            for tool_name in tool_names:
                if not isinstance(tool_name, str):
                    continue
                    
                if tool_name in available_tools:
                    valid_tools.append(tool_name)
                else:
                    self.logger.warning(f"Tool not found in registry: {tool_name}")
                    
            return valid_tools
            
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON response: {response}")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing tool names: {str(e)}")
            return []
    
    def can_handle(self, request) -> float:
        """
        Determine if this agent can handle the request.
        
        Args:
            request: Request object
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Check if the request is about finding tools
        prompt = request.prompt.lower()
        
        # Keywords that suggest tool finding
        tool_keywords = ["tool", "function", "capability", "what can you do", "help me with"]
        
        # Calculate confidence based on keyword presence
        confidence = 0.0
        for keyword in tool_keywords:
            if keyword in prompt:
                confidence += 0.2
                
        # Cap at 1.0
        return min(confidence, 1.0)
    
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
                template_id="tool_finder"
            )
            return prompt
        except ValueError:
            # Fallback to hardcoded prompt
            self.logger.warning("Tool Finder system prompt template not found, using fallback")
            return """You are a Tool Finder responsible for identifying relevant tools for user requests.
Your task is to analyze user requests and determine which tools would be most helpful.
Only recommend tools that are directly relevant to the request.
Return results as a simple JSON array of tool names.
""" 