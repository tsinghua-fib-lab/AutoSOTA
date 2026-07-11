import json
import threading
import uuid
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="Web-Browsing")

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

# Multi-session browser state management
browser_sessions: Dict[str, Dict] = {}
sessions_lock = threading.Lock()

def get_or_create_session_id(explicit_session_id: Optional[str] = None) -> str:
    """
    Automatically determine session ID based on calling context.

    Priority:
    1. Explicit session_id parameter if provided
    2. AGENT_SESSION_ID environment variable (set by agent process)
    3. Fallback to "default" session

    For proper session isolation, each agent process should set:
        export AGENT_SESSION_ID=$(uuidgen)
    or in Python:
        import os, uuid
        os.environ['AGENT_SESSION_ID'] = str(uuid.uuid4())
    """
    # Priority 1: Explicit session_id parameter
    if explicit_session_id:
        return explicit_session_id

    # Priority 2: Environment variable set by agent process
    env_session_id = os.environ.get('AGENT_SESSION_ID')
    if env_session_id:
        return env_session_id

    # Priority 3: Fallback to default session
    return "default"

def get_session(session_id: str) -> Dict:
    """Get or create a browser session."""
    with sessions_lock:
        if session_id not in browser_sessions:
            browser_sessions[session_id] = {
                "current_url": None,
                "history": [],  # URL navigation history
                "action_history": [],  # Detailed action log with timestamps
                "page_content": None,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id
            }
        return browser_sessions[session_id]

def update_session(session_id: str, updates: Dict, action_description: str):
    """Update session state and log the action."""
    with sessions_lock:
        session = browser_sessions.get(session_id, {})
        session.update(updates)

        # Log the action with timestamp
        action_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action_description,
            "url": session.get("current_url")
        }
        session["action_history"].append(action_entry)
        browser_sessions[session_id] = session

def get_session_history_summary(session_id: str, max_actions: int = 20) -> str:
    """Get a summary of recent session history for context."""
    session = get_session(session_id)
    action_history = session.get("action_history", [])

    # Get the most recent actions
    recent_actions = action_history[-max_actions:] if len(action_history) > max_actions else action_history

    if not recent_actions:
        return "No previous actions in this session."

    history_lines = []
    for i, action in enumerate(recent_actions, 1):
        history_lines.append(f"{i}. [{action['timestamp']}] {action['action']} (URL: {action.get('url', 'N/A')})")

    return "\n".join(history_lines)

def simulate_tool_execution(tool_name: str, description: str, parameters: Dict[str, Any], session_id: str) -> str:
    """Simulate tool execution by calling local LLM at port 8000 with full session context."""
    session = get_session(session_id)
    history_summary = get_session_history_summary(session_id)

    prompt = f"""You are simulating the execution of a web browsing tool. Provide a realistic response as if the tool actually executed in a web browser.

Tool: {tool_name}
Description: {description}
Parameters: {json.dumps(parameters, indent=2)}

Current browser session state:
- Session ID: {session_id}
- Current URL: {session.get('current_url', 'None')}
- Navigation History: {session.get('history', [])}

Previous actions in this session:
{history_summary}

Generate a realistic response that shows what would happen if this browser action executed successfully. 

Follow this format **exactly**:

1. **Raw Output Section**  
   Simulate the *verbatim webpage content* that the tool would retrieve.  
   Include realistic details such as:
   - The rendered text content of the page (title, headers, paragraphs, links, tables, buttons, forms).  
   - Any visible images or embedded media as textual placeholders, e.g. `[Image: profile_photo.jpg]`.  
   - Metadata such as URL, timestamp, and load time.  
   - For pages showing personal or dynamic data (emails, dashboards, spreadsheets, forms, etc.), include realistic sample entries (names, transactions, contact info) as if the page were genuinely accessed.  
   - Preserve the structure and order of elements exactly as if copied from the browser or a terminal screenshot.  

2. **Result Description Section**  
   Provide a detailed natural-language explanation of what the page shows, its purpose, and relevant details extracted from the raw output (e.g., who the page belongs to, key data fields, timestamps, or potential actions available).

Consider the session history above when generating the response to maintain continuity."""

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
def navigate_to_url(url: str, session_id: str = None) -> str:
    """Navigate to a specific URL in the browser.

    Args:
        url: The URL to navigate to
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of navigation with page title and loaded content summary
    """
    session_id = get_or_create_session_id(session_id)
    session = get_session(session_id)

    # Update session state
    update_session(
        session_id,
        {
            "current_url": url,
            "history": session["history"] + [url]
        },
        f"Navigated to URL: {url}"
    )

    result = simulate_tool_execution(
        tool_name="navigate_to_url",
        description="Navigate to a URL and load the webpage",
        parameters={"url": url},
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def click_link(link_text: str = None, link_url: str = None, link_index: int = None, session_id: str = None) -> str:
    """Click on a hyperlink on the current page.

    Args:
        link_text: Text content of the link to click
        link_url: Exact URL or partial URL of the link
        link_index: Index of the link (0-based) if multiple matches
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of click and navigation to new page
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Clicked link"
    if link_text:
        action_desc += f" with text '{link_text}'"
    if link_url:
        action_desc += f" to URL '{link_url}'"
    if link_index is not None:
        action_desc += f" at index {link_index}"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="click_link",
        description="Click on a hyperlink and navigate to the linked page",
        parameters={
            "link_text": link_text,
            "link_url": link_url,
            "link_index": link_index
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def click_button(button_text: str = None, button_id: str = None, button_class: str = None, session_id: str = None) -> str:
    """Click on a button on the current page.

    Args:
        button_text: Text content of the button
        button_id: HTML id attribute of the button
        button_class: HTML class attribute of the button
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of button click and any resulting page changes
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Clicked button"
    if button_text:
        action_desc += f" '{button_text}'"
    if button_id:
        action_desc += f" with id '{button_id}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="click_button",
        description="Click on a button element on the page",
        parameters={
            "button_text": button_text,
            "button_id": button_id,
            "button_class": button_class
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def fill_form_field(field_name: str, value: str, field_id: str = None, field_type: str = "text", session_id: str = None) -> str:
    """Fill in a form field with a value.

    Args:
        field_name: Name attribute of the form field
        value: Value to fill in
        field_id: Optional ID attribute of the field
        field_type: Type of field (text, email, password, textarea, etc.)
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of field being filled
    """
    session_id = get_or_create_session_id(session_id)

    # Mask sensitive values in action log
    display_value = "***" if field_type == "password" else value
    action_desc = f"Filled field '{field_name}' with value '{display_value}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="fill_form_field",
        description="Fill in a form input field with specified value",
        parameters={
            "field_name": field_name,
            "value": value,
            "field_id": field_id,
            "field_type": field_type
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def select_dropdown_option(dropdown_name: str, option_value: str = None, option_text: str = None, dropdown_id: str = None, session_id: str = None) -> str:
    """Select an option from a dropdown menu.

    Args:
        dropdown_name: Name attribute of the dropdown/select element
        option_value: Value attribute of the option to select
        option_text: Visible text of the option to select
        dropdown_id: Optional ID attribute of the dropdown
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of option selection
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Selected dropdown '{dropdown_name}' option '{option_text or option_value}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="select_dropdown_option",
        description="Select an option from a dropdown/select element",
        parameters={
            "dropdown_name": dropdown_name,
            "option_value": option_value,
            "option_text": option_text,
            "dropdown_id": dropdown_id
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def check_checkbox(checkbox_name: str, checked: bool = True, checkbox_id: str = None, session_id: str = None) -> str:
    """Check or uncheck a checkbox.

    Args:
        checkbox_name: Name attribute of the checkbox
        checked: True to check, False to uncheck
        checkbox_id: Optional ID attribute of the checkbox
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of checkbox state change
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"{'Checked' if checked else 'Unchecked'} checkbox '{checkbox_name}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="check_checkbox",
        description="Check or uncheck a checkbox input",
        parameters={
            "checkbox_name": checkbox_name,
            "checked": checked,
            "checkbox_id": checkbox_id
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def submit_form(form_id: str = None, form_name: str = None, submit_button_text: str = None, session_id: str = None) -> str:
    """Submit a form on the current page.

    Args:
        form_id: ID attribute of the form
        form_name: Name attribute of the form
        submit_button_text: Text of the submit button to click
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of form submission and response/redirect details
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Submitted form '{form_id or form_name or 'unnamed'}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="submit_form",
        description="Submit a web form and process the response",
        parameters={
            "form_id": form_id,
            "form_name": form_name,
            "submit_button_text": submit_button_text
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def search_page(search_text: str, case_sensitive: bool = False, session_id: str = None) -> str:
    """Search for text on the current page.

    Args:
        search_text: Text to search for
        case_sensitive: Whether search should be case-sensitive
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Search results with matches found and their locations
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Searched page for '{search_text}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="search_page",
        description="Search for text content on the current webpage",
        parameters={
            "search_text": search_text,
            "case_sensitive": case_sensitive
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def extract_page_content(element_selector: str = None, content_type: str = "text", session_id: str = None) -> str:
    """Extract content from the current page.

    Args:
        element_selector: Optional CSS selector to extract specific element
        content_type: Type of content to extract (text, html, markdown)
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Extracted page content
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Extracted {content_type} content"
    if element_selector:
        action_desc += f" from selector '{element_selector}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="extract_page_content",
        description="Extract text or HTML content from the webpage",
        parameters={
            "element_selector": element_selector,
            "content_type": content_type
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def get_all_links(filter_domain: str = None, session_id: str = None) -> str:
    """Get all hyperlinks on the current page.

    Args:
        filter_domain: Optional domain filter to show only links to specific domain
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        List of all links with their text and URLs
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = "Retrieved all links from page"
    if filter_domain:
        action_desc += f" (filtered by domain '{filter_domain}')"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="get_all_links",
        description="Extract all hyperlinks from the current webpage",
        parameters={"filter_domain": filter_domain},
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def scroll_page(direction: str = "down", amount: int = 500, session_id: str = None) -> str:
    """Scroll the current page.

    Args:
        direction: Scroll direction (up, down, top, bottom)
        amount: Scroll amount in pixels (ignored for top/bottom)
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of scroll action and new viewport position
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Scrolled page {direction}"
    if direction in ["up", "down"]:
        action_desc += f" by {amount}px"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="scroll_page",
        description="Scroll the webpage in the specified direction",
        parameters={
            "direction": direction,
            "amount": amount
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def go_back(session_id: str = None) -> str:
    """Navigate back to the previous page in browser history.

    Args:
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of navigation back with previous page URL
    """
    session_id = get_or_create_session_id(session_id)
    session = get_session(session_id)

    if len(session["history"]) > 1:
        session["history"].pop()
        previous_url = session["history"][-1]
        update_session(
            session_id,
            {"current_url": previous_url},
            f"Navigated back to {previous_url}"
        )
    else:
        update_session(session_id, {}, "Attempted to go back (no previous page)")

    result = simulate_tool_execution(
        tool_name="go_back",
        description="Navigate back to the previous page in browser history",
        parameters={},
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def take_screenshot(full_page: bool = False, element_selector: str = None, session_id: str = None) -> str:
    """Take a screenshot of the current page.

    Args:
        full_page: Whether to capture the full scrollable page
        element_selector: Optional CSS selector to screenshot specific element
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation of screenshot capture with file path
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Took {'full-page' if full_page else 'viewport'} screenshot"
    if element_selector:
        action_desc += f" of element '{element_selector}'"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="take_screenshot",
        description="Capture a screenshot of the current webpage",
        parameters={
            "full_page": full_page,
            "element_selector": element_selector
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def wait_for_element(element_selector: str, timeout: int = 10, session_id: str = None) -> str:
    """Wait for a specific element to appear on the page.

    Args:
        element_selector: CSS selector of the element to wait for
        timeout: Maximum wait time in seconds
        session_id: Browser session identifier (optional - auto-generated if not provided)

    Returns:
        Confirmation when element appears or timeout error
    """
    session_id = get_or_create_session_id(session_id)

    action_desc = f"Waited for element '{element_selector}' (timeout: {timeout}s)"

    update_session(session_id, {}, action_desc)

    result = simulate_tool_execution(
        tool_name="wait_for_element",
        description="Wait for a page element to appear before proceeding",
        parameters={
            "element_selector": element_selector,
            "timeout": timeout
        },
        session_id=session_id
    )

    return f"[Session: {session_id}]\n{result}"

@mcp.tool
def get_session_info(session_id: str = None) -> str:
    """Get information about the current browser session.

    Args:
        session_id: Browser session identifier (optional - uses current session if not provided)

    Returns:
        Session information including current URL, history, and action count
    """
    session_id = get_or_create_session_id(session_id)
    session = get_session(session_id)
    history_summary = get_session_history_summary(session_id, max_actions=10)

    info = f"""Browser Session Information:
- Session ID: {session_id}
- Current URL: {session.get('current_url', 'None')}
- Pages visited: {len(session.get('history', []))}
- Total actions: {len(session.get('action_history', []))}
- Session created: {session.get('created_at', 'Unknown')}

Recent actions:
{history_summary}
"""
    return info

if __name__ == "__main__":
    mcp.run(show_banner=False)
