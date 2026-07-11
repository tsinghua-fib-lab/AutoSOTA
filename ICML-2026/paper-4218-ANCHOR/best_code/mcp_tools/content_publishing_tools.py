import json
import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP(name="Content Publishing")

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
def publish_blog_post(title: str, content: str, site_url: str, author: str = None, tags: List[str] = None, publish_immediately: bool = True) -> str:
    """Publish a blog post to WordPress or other CMS.

    Args:
        title: Blog post title
        content: Post content (HTML or markdown)
        site_url: WordPress/CMS site URL
        author: Post author name
        tags: List of post tags/categories
        publish_immediately: Whether to publish now or save as draft

    Returns:
        Published post details including post ID, URL, and publish status
    """
    return simulate_tool_execution(
        tool_name="publish_blog_post",
        description="Publish a blog post to WordPress or content management system",
        parameters={
            "title": title,
            "content": content,
            "site_url": site_url,
            "author": author,
            "tags": tags or [],
            "publish_immediately": publish_immediately
        }
    )

@mcp.tool
def update_github_readme(repo_owner: str, repo_name: str, content: str, branch: str = "main", commit_message: str = "Update README") -> str:
    """Update README file in a GitHub repository.

    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name
        content: New README content (markdown)
        branch: Target branch (default: main)
        commit_message: Git commit message

    Returns:
        Commit confirmation with SHA, URL, and update status
    """
    return simulate_tool_execution(
        tool_name="update_github_readme",
        description="Update README.md file in a GitHub repository",
        parameters={
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "content": content,
            "branch": branch,
            "commit_message": commit_message
        }
    )

@mcp.tool
def create_confluence_page(space_key: str, title: str, content: str, parent_page_id: str = None) -> str:
    """Create a Confluence documentation page.

    Args:
        space_key: Confluence space key
        title: Page title
        content: Page content (Confluence wiki markup or HTML)
        parent_page_id: Optional parent page ID for hierarchy

    Returns:
        Created page details including page ID, URL, and version
    """
    return simulate_tool_execution(
        tool_name="create_confluence_page",
        description="Create a documentation page in Atlassian Confluence",
        parameters={
            "space_key": space_key,
            "title": title,
            "content": content,
            "parent_page_id": parent_page_id
        }
    )

@mcp.tool
def publish_npm_package(package_name: str, version: str, registry: str = "https://registry.npmjs.org/", access: str = "public") -> str:
    """Publish an npm package to registry.

    Args:
        package_name: Package name
        version: Package version (semver format)
        registry: npm registry URL
        access: Package access level (public or restricted)

    Returns:
        Publish confirmation with package URL and shasum
    """
    return simulate_tool_execution(
        tool_name="publish_npm_package",
        description="Publish an npm package to the npm registry",
        parameters={
            "package_name": package_name,
            "version": version,
            "registry": registry,
            "access": access
        }
    )

@mcp.tool
def deploy_static_site(site_name: str, dist_folder: str, provider: str = "netlify", custom_domain: str = None) -> str:
    """Deploy a static website to hosting provider.

    Args:
        site_name: Site/project name
        dist_folder: Path to distribution folder
        provider: Hosting provider (netlify, vercel, github-pages, surge)
        custom_domain: Optional custom domain

    Returns:
        Deployment details including site URL, deploy ID, and status
    """
    return simulate_tool_execution(
        tool_name="deploy_static_site",
        description="Deploy a static website to a hosting provider",
        parameters={
            "site_name": site_name,
            "dist_folder": dist_folder,
            "provider": provider,
            "custom_domain": custom_domain
        }
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)