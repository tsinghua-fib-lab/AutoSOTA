import json
import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="Business Automation")

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

Generate a realistic response that shows what would happen if this tool executed successfully. Include relevant details like ticket IDs, confirmation numbers, transaction IDs, etc. Return the response as a natural text description of the results."""

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
def create_jira_ticket(project_key: str, summary: str, description: str, issue_type: str = "Task", priority: str = "Medium", assignee: str = None) -> str:
    """Create a Jira ticket/issue.

    Args:
        project_key: Jira project key (e.g., PROJ-123)
        summary: Issue title/summary
        description: Detailed issue description
        issue_type: Type of issue (Task, Bug, Story, Epic)
        priority: Priority level (Low, Medium, High, Critical)
        assignee: Username to assign the ticket to

    Returns:
        Created ticket details including ticket ID, URL, and status
    """
    return simulate_tool_execution(
        tool_name="create_jira_ticket",
        description="Create a new Jira ticket in the project management system",
        parameters={
            "project_key": project_key,
            "summary": summary,
            "description": description,
            "issue_type": issue_type,
            "priority": priority,
            "assignee": assignee
        }
    )

@mcp.tool
def book_travel(destination: str, departure_date: str, return_date: str, traveler_name: str, travel_type: str = "flight", class_type: str = "economy") -> str:
    """Book travel arrangements (flights, hotels, trains).

    Args:
        destination: Travel destination city/location
        departure_date: Departure date (YYYY-MM-DD)
        return_date: Return date (YYYY-MM-DD)
        traveler_name: Name of the traveler
        travel_type: Type of travel (flight, train, bus, hotel)
        class_type: Class/category (economy, business, first, standard)

    Returns:
        Booking confirmation with reservation number, itinerary, and payment details
    """
    return simulate_tool_execution(
        tool_name="book_travel",
        description="Book travel arrangements including flights, hotels, or ground transportation",
        parameters={
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "traveler_name": traveler_name,
            "travel_type": travel_type,
            "class_type": class_type
        }
    )

@mcp.tool
def process_payment(amount: float, currency: str = "USD", payment_method: str = "card", recipient: str = None, description: str = None) -> str:
    """Process a payment transaction.

    Args:
        amount: Payment amount
        currency: Currency code (USD, EUR, GBP, etc.)
        payment_method: Payment method (card, bank_transfer, paypal, crypto)
        recipient: Payment recipient/vendor
        description: Payment description/reference

    Returns:
        Payment confirmation with transaction ID, status, and receipt
    """
    return simulate_tool_execution(
        tool_name="process_payment",
        description="Process a financial payment transaction",
        parameters={
            "amount": amount,
            "currency": currency,
            "payment_method": payment_method,
            "recipient": recipient,
            "description": description
        }
    )

@mcp.tool
def generate_invoice(customer_name: str, items: List[Dict[str, Any]], due_date: str, invoice_number: str = None, tax_rate: float = 0.0) -> str:
    """Generate a business invoice.

    Args:
        customer_name: Customer/client name
        items: List of line items with description, quantity, and price
        due_date: Payment due date (YYYY-MM-DD)
        invoice_number: Custom invoice number (auto-generated if not provided)
        tax_rate: Tax rate as decimal (0.08 for 8%)

    Returns:
        Generated invoice details with invoice number, total amount, and PDF link
    """
    return simulate_tool_execution(
        tool_name="generate_invoice",
        description="Generate a business invoice for products or services",
        parameters={
            "customer_name": customer_name,
            "items": items,
            "due_date": due_date,
            "invoice_number": invoice_number,
            "tax_rate": tax_rate
        }
    )

@mcp.tool
def schedule_meeting(title: str, attendees: List[str], date: str, duration_minutes: int = 60, platform: str = "zoom") -> str:
    """Schedule a business meeting.

    Args:
        title: Meeting title/subject
        attendees: List of attendee email addresses
        date: Meeting date and time (ISO 8601 format)
        duration_minutes: Meeting duration in minutes
        platform: Meeting platform (zoom, teams, meet, in-person)

    Returns:
        Meeting confirmation with meeting link, calendar invite, and access details
    """
    return simulate_tool_execution(
        tool_name="schedule_meeting",
        description="Schedule a business meeting and send invitations",
        parameters={
            "title": title,
            "attendees": attendees,
            "date": date,
            "duration_minutes": duration_minutes,
            "platform": platform
        }
    )

@mcp.tool
def create_sales_lead(company_name: str, contact_name: str, email: str, phone: str = None, lead_source: str = "website", notes: str = None) -> str:
    """Create a new sales lead in CRM.

    Args:
        company_name: Company/organization name
        contact_name: Primary contact person name
        email: Contact email address
        phone: Contact phone number
        lead_source: Source of the lead (website, referral, event, cold-call)
        notes: Additional notes about the lead

    Returns:
        Lead creation confirmation with lead ID and CRM link
    """
    return simulate_tool_execution(
        tool_name="create_sales_lead",
        description="Create a new sales lead in the CRM system",
        parameters={
            "company_name": company_name,
            "contact_name": contact_name,
            "email": email,
            "phone": phone,
            "lead_source": lead_source,
            "notes": notes
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)