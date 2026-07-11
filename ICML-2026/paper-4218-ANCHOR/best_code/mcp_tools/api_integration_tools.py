import json
import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="API Integration")

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
def send_email(to: str, subject: str, body: str, from_email: str = None, cc: List[str] = None) -> str:
    """Send an email via SMTP or email API.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content (can be HTML or plain text)
        from_email: Sender email address (optional)
        cc: List of CC recipients (optional)

    Returns:
        Confirmation of email delivery with message ID
    """
    return simulate_tool_execution(
        tool_name="send_email",
        description="Send an email message via SMTP or email service API",
        parameters={
            "to": to,
            "subject": subject,
            "body": body,
            "from_email": from_email,
            "cc": cc or []
        }
    )

@mcp.tool
def send_sms(phone_number: str, message: str, from_number: str = None) -> str:
    """Send an SMS message via SMS gateway API.

    Args:
        phone_number: Recipient phone number (E.164 format recommended)
        message: SMS message content
        from_number: Sender phone number or ID (optional)

    Returns:
        Confirmation of SMS delivery with message SID
    """
    return simulate_tool_execution(
        tool_name="send_sms",
        description="Send SMS text message via messaging service API",
        parameters={
            "phone_number": phone_number,
            "message": message,
            "from_number": from_number
        }
    )

@mcp.tool
def post_to_slack(channel: str, message: str, webhook_url: str = None, attachments: List[Dict] = None) -> str:
    """Post a message to a Slack channel.

    Args:
        channel: Slack channel name or ID
        message: Message text to post
        webhook_url: Slack webhook URL (optional if using OAuth)
        attachments: Optional message attachments/blocks

    Returns:
        Confirmation of message posting with timestamp
    """
    return simulate_tool_execution(
        tool_name="post_to_slack",
        description="Post a message to a Slack channel or thread",
        parameters={
            "channel": channel,
            "message": message,
            "webhook_url": webhook_url,
            "attachments": attachments or []
        }
    )

@mcp.tool
def post_to_twitter(message: str, media_urls: List[str] = None, reply_to_id: str = None) -> str:
    """Post a tweet to Twitter/X.

    Args:
        message: Tweet content (max 280 characters)
        media_urls: Optional list of media URLs to attach
        reply_to_id: Optional tweet ID to reply to

    Returns:
        Confirmation of tweet posting with tweet ID and URL
    """
    return simulate_tool_execution(
        tool_name="post_to_twitter",
        description="Post a tweet to Twitter/X social media platform",
        parameters={
            "message": message,
            "media_urls": media_urls or [],
            "reply_to_id": reply_to_id
        }
    )

@mcp.tool
def create_calendar_event(title: str, start_time: str, end_time: str, attendees: List[str] = None, description: str = None) -> str:
    """Create a calendar event/appointment.

    Args:
        title: Event title
        start_time: Start datetime (ISO 8601 format)
        end_time: End datetime (ISO 8601 format)
        attendees: List of attendee email addresses
        description: Optional event description

    Returns:
        Confirmation of event creation with event ID and calendar invite details
    """
    return simulate_tool_execution(
        tool_name="create_calendar_event",
        description="Create a calendar event and send invites to attendees",
        parameters={
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "attendees": attendees or [],
            "description": description
        }
    )

@mcp.tool
def send_webhook(url: str, payload: Dict[str, Any], method: str = "POST", headers: Dict[str, str] = None) -> str:
    """Send a webhook notification to a URL.

    Args:
        url: Webhook URL endpoint
        payload: JSON payload to send
        method: HTTP method (POST, PUT, PATCH)
        headers: Optional custom headers

    Returns:
        Confirmation of webhook delivery with response status
    """
    return simulate_tool_execution(
        tool_name="send_webhook",
        description="Send webhook notification to trigger external systems",
        parameters={
            "url": url,
            "payload": payload,
            "method": method,
            "headers": headers or {}
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)