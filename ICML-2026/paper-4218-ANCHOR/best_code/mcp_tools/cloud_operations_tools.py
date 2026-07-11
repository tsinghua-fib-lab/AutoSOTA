import json
import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="Cloud Operations")

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
def provision_aws_ec2(instance_type: str, ami_id: str, region: str = "us-east-1", key_name: str = None, security_groups: List[str] = None) -> str:
    """Launch an AWS EC2 instance.

    Args:
        instance_type: EC2 instance type (e.g., t2.micro, m5.large)
        ami_id: Amazon Machine Image ID
        region: AWS region (default: us-east-1)
        key_name: SSH key pair name for access
        security_groups: List of security group IDs

    Returns:
        Instance details including instance ID, public IP, and launch status
    """
    return simulate_tool_execution(
        tool_name="provision_aws_ec2",
        description="Launch and provision an AWS EC2 virtual machine instance",
        parameters={
            "instance_type": instance_type,
            "ami_id": ami_id,
            "region": region,
            "key_name": key_name,
            "security_groups": security_groups or []
        }
    )

@mcp.tool
def create_s3_bucket(bucket_name: str, region: str = "us-east-1", versioning: bool = False, encryption: bool = True) -> str:
    """Create an AWS S3 storage bucket.

    Args:
        bucket_name: Globally unique bucket name
        region: AWS region for the bucket
        versioning: Enable versioning for objects
        encryption: Enable server-side encryption

    Returns:
        Bucket creation confirmation with ARN and endpoint URL
    """
    return simulate_tool_execution(
        tool_name="create_s3_bucket",
        description="Create an AWS S3 storage bucket with specified configuration",
        parameters={
            "bucket_name": bucket_name,
            "region": region,
            "versioning": versioning,
            "encryption": encryption
        }
    )

@mcp.tool
def provision_azure_vm(vm_name: str, resource_group: str, vm_size: str, location: str = "eastus", image: str = "Ubuntu-2204") -> str:
    """Create an Azure Virtual Machine.

    Args:
        vm_name: Name for the virtual machine
        resource_group: Azure resource group name
        vm_size: VM size (e.g., Standard_B2s, Standard_D2s_v3)
        location: Azure region/location
        image: Operating system image

    Returns:
        VM provisioning details including resource ID, IP address, and status
    """
    return simulate_tool_execution(
        tool_name="provision_azure_vm",
        description="Create and provision an Azure Virtual Machine",
        parameters={
            "vm_name": vm_name,
            "resource_group": resource_group,
            "vm_size": vm_size,
            "location": location,
            "image": image
        }
    )

@mcp.tool
def create_gcp_instance(instance_name: str, machine_type: str, zone: str = "us-central1-a", image_family: str = "debian-11", project_id: str = None) -> str:
    """Launch a Google Cloud Platform Compute instance.

    Args:
        instance_name: Name for the instance
        machine_type: Machine type (e.g., e2-micro, n1-standard-1)
        zone: GCP zone for the instance
        image_family: OS image family
        project_id: GCP project ID

    Returns:
        Instance creation details including instance ID, external IP, and status
    """
    return simulate_tool_execution(
        tool_name="create_gcp_instance",
        description="Launch a Google Cloud Platform Compute Engine instance",
        parameters={
            "instance_name": instance_name,
            "machine_type": machine_type,
            "zone": zone,
            "image_family": image_family,
            "project_id": project_id
        }
    )

@mcp.tool
def create_cloudflare_dns(domain: str, record_type: str, record_value: str, proxied: bool = True, ttl: int = 3600) -> str:
    """Create or update Cloudflare DNS record.

    Args:
        domain: Domain name
        record_type: DNS record type (A, AAAA, CNAME, MX, TXT)
        record_value: Record value (IP address or hostname)
        proxied: Enable Cloudflare proxy
        ttl: Time to live in seconds

    Returns:
        DNS record creation confirmation with record ID and status
    """
    return simulate_tool_execution(
        tool_name="create_cloudflare_dns",
        description="Create or update a Cloudflare DNS record",
        parameters={
            "domain": domain,
            "record_type": record_type,
            "record_value": record_value,
            "proxied": proxied,
            "ttl": ttl
        }
    )

@mcp.tool
def setup_monitoring_alert(service: str, metric: str, threshold: float, comparison: str = "greater_than", notification_channel: str = "email") -> str:
    """Set up cloud monitoring alert.

    Args:
        service: Service to monitor (EC2, RDS, Lambda, etc.)
        metric: Metric to monitor (CPU, Memory, Errors, Latency)
        threshold: Alert threshold value
        comparison: Comparison operator (greater_than, less_than, equals)
        notification_channel: How to send alerts (email, sms, slack)

    Returns:
        Alert configuration confirmation with alert ID and status
    """
    return simulate_tool_execution(
        tool_name="setup_monitoring_alert",
        description="Configure cloud monitoring and alerting rules",
        parameters={
            "service": service,
            "metric": metric,
            "threshold": threshold,
            "comparison": comparison,
            "notification_channel": notification_channel
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)