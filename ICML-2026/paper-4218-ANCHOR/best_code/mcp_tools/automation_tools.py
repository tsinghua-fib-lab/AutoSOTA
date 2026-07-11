import json
import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="DevOps Automation")

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

{SHARED_PROMPT}

"""

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
def trigger_ci_build(project_name: str, branch: str = "main", parameters: Dict[str, str] = None, ci_system: str = "jenkins") -> str:
    """Trigger a CI/CD build pipeline.

    Args:
        project_name: Name of the project/job
        branch: Git branch to build
        parameters: Optional build parameters
        ci_system: CI system to use (jenkins, gitlab, github-actions, circleci)

    Returns:
        Build trigger confirmation with build ID, queue position, and pipeline URL
    """
    return simulate_tool_execution(
        tool_name="trigger_ci_build",
        description="Trigger a CI/CD build pipeline execution",
        parameters={
            "project_name": project_name,
            "branch": branch,
            "parameters": parameters or {},
            "ci_system": ci_system
        }
    )

@mcp.tool
def deploy_docker_container(image: str, container_name: str, port_mappings: Dict[int, int] = None, environment_vars: Dict[str, str] = None, host: str = "localhost") -> str:
    """Deploy and run a Docker container.

    Args:
        image: Docker image name with tag
        container_name: Name for the container
        port_mappings: Port mappings (container_port: host_port)
        environment_vars: Environment variables to set
        host: Docker host to deploy to

    Returns:
        Container deployment details including container ID, status, and exposed ports
    """
    return simulate_tool_execution(
        tool_name="deploy_docker_container",
        description="Deploy and run a Docker container on specified host",
        parameters={
            "image": image,
            "container_name": container_name,
            "port_mappings": port_mappings or {},
            "environment_vars": environment_vars or {},
            "host": host
        }
    )

@mcp.tool
def create_kubernetes_deployment(deployment_name: str, image: str, replicas: int = 1, namespace: str = "default", environment_vars: Dict[str, str] = None) -> str:
    """Create a Kubernetes deployment.

    Args:
        deployment_name: Name of the deployment
        image: Container image with tag
        replicas: Number of pod replicas
        namespace: Kubernetes namespace
        environment_vars: Environment variables for containers

    Returns:
        Deployment details including status, pod count, and service endpoints
    """
    return simulate_tool_execution(
        tool_name="create_kubernetes_deployment",
        description="Create and deploy a Kubernetes deployment with pods",
        parameters={
            "deployment_name": deployment_name,
            "image": image,
            "replicas": replicas,
            "namespace": namespace,
            "environment_vars": environment_vars or {}
        }
    )

@mcp.tool
def order_food_delivery(restaurant: str, items: List[Dict[str, Any]], delivery_address: str, delivery_time: str = "ASAP", payment_method: str = "card") -> str:
    """Order food for delivery from a restaurant.

    Args:
        restaurant: Restaurant name or ID
        items: List of menu items with quantities
        delivery_address: Delivery address
        delivery_time: Requested delivery time or "ASAP"
        payment_method: Payment method (card, cash, app)

    Returns:
        Order confirmation with order ID, estimated delivery time, and tracking info
    """
    return simulate_tool_execution(
        tool_name="order_food_delivery",
        description="Place a food delivery order from a restaurant",
        parameters={
            "restaurant": restaurant,
            "items": items,
            "delivery_address": delivery_address,
            "delivery_time": delivery_time,
            "payment_method": payment_method
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)