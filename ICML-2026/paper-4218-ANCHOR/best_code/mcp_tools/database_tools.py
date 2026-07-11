import json
import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="Database Operations")

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
def sql_execute(query: str, database: str, parameters: Dict[str, Any] = None) -> str:
    """Execute SQL INSERT, UPDATE, or DELETE operations on a database.

    Args:
        query: The SQL query to execute (INSERT, UPDATE, DELETE)
        database: Database connection string or name
        parameters: Optional query parameters for prepared statements

    Returns:
        Execution result with affected rows and details
    """
    return simulate_tool_execution(
        tool_name="sql_execute",
        description="Execute SQL INSERT/UPDATE/DELETE operations on a database",
        parameters={
            "query": query,
            "database": database,
            "parameters": parameters or {}
        }
    )

@mcp.tool
def create_database_backup(database_name: str, backup_location: str, backup_type: str = "full", compression: bool = True) -> str:
    """Create a database backup.

    Args:
        database_name: Name of the database to backup
        backup_location: Storage location for backup file
        backup_type: Type of backup (full, incremental, differential)
        compression: Enable backup compression

    Returns:
        Backup confirmation with file path, size, and timestamp
    """
    return simulate_tool_execution(
        tool_name="create_database_backup",
        description="Create a backup of the specified database",
        parameters={
            "database_name": database_name,
            "backup_location": backup_location,
            "backup_type": backup_type,
            "compression": compression
        }
    )

@mcp.tool
def mongodb_operation(collection: str, operation: str, document: Dict[str, Any] = None, filter: Dict[str, Any] = None, database: str = "default") -> str:
    """Perform MongoDB operations.

    Args:
        collection: Collection name
        operation: Operation type (insert, update, delete, find)
        document: Document to insert or update
        filter: Query filter for find/update/delete
        database: Database name

    Returns:
        Operation result with affected documents and IDs
    """
    return simulate_tool_execution(
        tool_name="mongodb_operation",
        description="Perform MongoDB database operations",
        parameters={
            "collection": collection,
            "operation": operation,
            "document": document or {},
            "filter": filter or {},
            "database": database
        }
    )

@mcp.tool
def redis_command(command: str, key: str, value: Any = None, ttl: int = None) -> str:
    """Execute Redis cache operations.

    Args:
        command: Redis command (SET, GET, DEL, EXPIRE, HSET, etc.)
        key: Redis key
        value: Value to set (for SET operations)
        ttl: Time to live in seconds

    Returns:
        Command execution result
    """
    return simulate_tool_execution(
        tool_name="redis_command",
        description="Execute Redis cache operations",
        parameters={
            "command": command,
            "key": key,
            "value": value,
            "ttl": ttl
        }
    )

@mcp.tool
def create_database_user(username: str, password: str, database_name: str, privileges: List[str] = None, host: str = "localhost") -> str:
    """Create a new database user with specified privileges.

    Args:
        username: New username
        password: User password
        database_name: Database to grant access to
        privileges: List of privileges (SELECT, INSERT, UPDATE, DELETE, etc.)
        host: Host access pattern

    Returns:
        User creation confirmation with granted privileges
    """
    return simulate_tool_execution(
        tool_name="create_database_user",
        description="Create a new database user with specified privileges",
        parameters={
            "username": username,
            "password": password,
            "database_name": database_name,
            "privileges": privileges or ["SELECT"],
            "host": host
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)