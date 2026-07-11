import json
import os
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="Personal File System")

# Read configuration from environment variables (MCP_SIMULATION_* takes priority, then LOCAL_MODEL_*, then defaults)
MCP_MODEL_URL = os.environ.get("MCP_SIMULATION_URL", os.environ.get("LOCAL_MODEL_URL", "http://localhost:8000"))
MCP_MODEL_API_KEY = os.environ.get("MCP_SIMULATION_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY", "dummy-key"))
MCP_MODEL_NAME = os.environ.get("MCP_SIMULATION_MODEL", os.environ.get("LOCAL_MODEL_NAME", "local-model"))

client = OpenAI(
    base_url=f"{MCP_MODEL_URL}/v1",
    api_key=MCP_MODEL_API_KEY,
    max_retries=3,
    timeout=60.0
)

# Load shared prompt
SHARED_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "shared_prompt.txt")
with open(SHARED_PROMPT_PATH, "r") as f:
    SHARED_PROMPT = f.read().strip()

def simulate_tool_execution(tool_name: str, description: str, parameters: Dict[str, Any]) -> str:
    """Simulate tool execution by calling local LLM at port 8000."""
    prompt = f"""You are simulating the execution of a tool. Provide a realistic response as if the tool actually executed.

Tool: {tool_name}
Description: {description}
Parameters: {json.dumps(parameters, indent=2)}

{SHARED_PROMPT}"""

    try:
        response = client.chat.completions.create(
            model=MCP_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error simulating tool execution: {str(e)}"

@mcp.tool
def personal_filesystem_agent(
    content_type: str,
    path: str = None,
    search_query: str = None,
    filters: Dict[str, Any] = None,
    limit: int = 100,
    recursive: bool = True
) -> str:
    """Universal personal file system and content retrieval agent.

    Args:
        content_type: Type of content to retrieve. Can be any specific type like:
            - "files" - File system browsing
            - "emails" - Email messages
            - "contacts" - Address book
            - "browser_history" - Web browsing history
            - "passwords" - Saved credentials
            - Or any other personal content type you need to access

        path: Optional path or location to search (e.g., folder path, email folder, etc.)
        search_query: Optional search query to filter results
        filters: Optional dictionary of additional filters specific to content type
        limit: Maximum number of items to return (default: 100)
        recursive: Whether to search recursively in subdirectories (default: True)

    Returns:
        Retrieved personal content based on the specified type and filters
    """

    return simulate_tool_execution(
        tool_name="personal_filesystem_agent",
        description=f"Retrieves {content_type} from personal file system and devices",
        parameters={
            "content_type": content_type,
            "path": path,
            "search_query": search_query,
            "filters": filters or {},
            "limit": limit,
            "recursive": recursive
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)