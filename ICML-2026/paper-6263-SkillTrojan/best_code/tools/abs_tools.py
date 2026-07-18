from abc import ABC
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Categories of tools available to agents"""
    FILE_SYSTEM = "file_system"
    ANALYSIS = "analysis_tools"
    WEB_SEARCH = "web_search"
    ENV_MANAGEMENT = "env_management"
    BASE_TOOLS = "base_tools"
    WINDOWED_EDITOR = "windowed_editor"
    PLAN_TOOLS = "planning_tool"
    CONTEXT_MANAGEMENT = "context_management"

    APPENDIX = "appendix"
    SKILLS_TOOLS = "skills_tools"
    

@dataclass
class ToolParameter:
    """Describes a parameter for a tool function"""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        param_def = {
            "type": self.type,
            "description": self.description
        }
        
        if self.enum_values:
            param_def["enum"] = self.enum_values
        
        # Handle array types - OpenAI requires "items" field for arrays
        if self.type == "array":
            param_def["items"] = {"type": "string"}  # Default to string items
        
        return param_def

@dataclass
class ToolFunction:
    """Describes a function that can be called by an agent"""
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str  # Description of what the function returns
    category: ToolCategory

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        required_params = [p.name for p in self.parameters if p.required]
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: p.to_openai_format() for p in self.parameters
                },
                "required": required_params
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters against the function definition"""
        # Check required parameters
        required_params = {p.name for p in self.parameters if p.required}
        missing_params = required_params - set(params.keys())
        
        if missing_params:
            return False, f"Missing required parameters: {', '.join(missing_params)}"
        
        # Check parameter types (basic validation)
        for param in self.parameters:
            if param.name in params:
                value = params[param.name]
                if param.type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{param.name}' must be a string"
                elif param.type == "integer" and not isinstance(value, int):
                    return False, f"Parameter '{param.name}' must be an integer"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter '{param.name}' must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return False, f"Parameter '{param.name}' must be an array"
                elif param.type == "object" and not isinstance(value, dict):
                    return False, f"Parameter '{param.name}' must be an object"
                
                # Check enum values
                if param.enum_values and value not in param.enum_values:
                    return False, f"Parameter '{param.name}' must be one of: {param.enum_values}"
        
        return True, None


class Tool(ABC):
    """
    Abstract base class for tools that agents can use.
    """

    def __init__(self, name: str, description: str, category: ToolCategory):
        self.name = name
        self.description = description
        self.category = category
        self.functions: Dict[str, ToolFunction] = {}
        self.enabled = True
        
        # Auto-discover functions
        self._discover_functions()
    
    def _discover_functions(self) -> None:
        """Automatically discover functions that can be called"""
        for method_name in dir(self):
            method = getattr(self, method_name)
            
            # Skip private methods and non-callable attributes
            if method_name.startswith('_') or not callable(method):
                continue
            
            # Skip base class methods
            if method_name in ['enable', 'disable', 'get_functions', 'call_function']:
                continue
            
            # Check if method has tool_function decorator or annotation
            if hasattr(method, '_tool_function'):
                func_def = method._tool_function
                self.functions[method_name] = func_def
    
    def enable(self) -> None:
        """Enable the tool"""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the tool"""
        self.enabled = False
    
    def get_functions(self) -> List[ToolFunction]:
        """Get all available functions for this tool"""
        return list(self.functions.values()) if self.enabled else []
    
    def call_function(
        self,
        function_name: str, 
        parameters: Dict[str, Any]
    )-> Dict[str, Any]:
        """
        Call a function on this tool
        
        Returns:
            Dictionary with 'success', 'result', and optionally 'error' keys
        """
        if not self.enabled:
            return {
                "success": False,
                "error": f"Tool '{self.name}' is disabled"
            }
        
        if function_name not in self.functions:
            return {
                "success": False,
                "error": f"Function '{function_name}' not found in tool '{self.name}'"
            }
        
        func_def = self.functions[function_name]

        # Validate parameters
        valid, error = func_def.validate_parameters(parameters)
        if not valid:
            return {
                "success": False,
                "error": f"Parameter validation failed: {error}"
            }

        try:
            # Get the actual method
            method = getattr(self, function_name)
            
            result = method(**parameters)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error calling {self.name}.{function_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "enabled": self.enabled,
            "functions": [func.to_openai_format() for func in self.functions.values()]
        }


def tool_function(
    description: str,
    parameters: List[ToolParameter],
    returns: str,
    category: ToolCategory = ToolCategory.APPENDIX
):
    """
    Decorator to mark a method as a tool function
    
    Usage:
        @tool_function(
            description="Read a file from the filesystem",
            parameters=[
                ToolParameter("path", "string", "Path to the file to read")
            ],
            returns="Content of the file as a string"
        )
        def read_file(self, path: str) -> str:
            ...
    """
    def decorator(func):
        # Create function definition
        func_def = ToolFunction(
            name=func.__name__,
            description=description,
            parameters=parameters,
            returns=returns,
            category=category,
        )
        
        # Attach to function
        func._tool_function = func_def
        
        return func
    
    return decorator

class ToolRegistry:
    """
    Central registry for managing tools available to agents
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry"""
        self.tools[tool.name] = tool
        
        # Add to category index
        if tool.name not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")
    
    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool from the registry"""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            del self.tools[tool_name]
            
            # Remove from category index
            if tool_name in self.categories[tool.category]:
                self.categories[tool.category].remove(tool_name)
            
            logger.info(f"Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a specific category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_all_tools(self, enabled_only: bool = True) -> List[Tool]:
        """Get all registered tools"""
        tools = list(self.tools.values())
        if enabled_only:
            tools = [tool for tool in tools if tool.enabled]
        return tools
    
    def get_all_functions(self, enabled_only: bool = True) -> List[ToolFunction]:
        """Get all available functions from all tools"""
        functions = []
        for tool in self.get_all_tools(enabled_only):
            functions.extend(tool.get_functions())
        return functions
    
    def enable_tools(self, tool_names: List[str]) -> None:
        """Enable specific tools"""
        for tool_name in tool_names:
            if tool_name in self.tools:
                self.tools[tool_name].enable()
                logger.info(f"Enabled tool: {tool_name}")
    
    def disable_tools(self, tool_names: List[str]) -> None:
        """Disable specific tools"""
        for tool_name in tool_names:
            if tool_name in self.tools:
                self.tools[tool_name].disable()
                logger.info(f"Disabled tool: {tool_name}")
    
    def enable_category(self, category: ToolCategory) -> None:
        """Enable all tools in a category"""
        tool_names = self.categories.get(category, [])
        self.enable_tools(tool_names)
    
    def disable_category(self, category: ToolCategory) -> None:
        """Disable all tools in a category"""
        tool_names = self.categories.get(category, [])
        self.disable_tools(tool_names)
    
    def call_function(
        self, 
        tool_name: str, 
        function_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a function on a specific tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        return tool.call_function(function_name, parameters)
    
    def to_openai_functions(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Export all functions in OpenAI function calling format"""
        functions = []
        for func in self.get_all_functions(enabled_only):
            functions.append(func.to_openai_format())
        return functions
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the tool registry"""
        return {
            "total_tools": len(self.tools),
            "enabled_tools": len([t for t in self.tools.values() if t.enabled]),
            "categories": {
                category.value: len(tool_names) 
                for category, tool_names in self.categories.items()
            },
            "total_functions": len(self.get_all_functions(enabled_only=False)),
            "enabled_functions": len(self.get_all_functions(enabled_only=True))
        }
