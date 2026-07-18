"""
Context Management Tools for SafeFlow - Plan Recording and Memory Management

Provides specialized functions for ContextManagerAgent:
- Plan recording and tracking from DefaultAgent
- Memory management and conversation history
- Simple coordination without overlapping with BaseTools functionality

Focused on plan management rather than workspace initialization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from tools.abs_tools import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class ContextTools(Tool):
    """
    Context Management Tools for SafeFlow.

    Specialized tools for ContextManagerAgent to record plans and manage memory.
    Does not overlap with BaseTools initialization functionality.
    """

    def __init__(
        self,
        item_id: str,
        name: str = "context_tools",
        description: str = "Plan recording and memory management for ContextManagerAgent",
    ):
        super().__init__(
            name=name,
            description=description,
            category=ToolCategory.CONTEXT_MANAGEMENT
        )

        self.item_id = item_id

    def _validate_absolute_path(self, path: str) -> Path:
        """Validate absolute path and return resolved Path object."""
        p = Path(path)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute, got: {path}")
        return p.resolve()

    @tool_function(
        description="Record a plan created by DefaultAgent for tracking and memory.",
        parameters=[
            ToolParameter("plan_content", "string", "The plan content from DefaultAgent", required=True),
            ToolParameter("plan_type", "string", "Type of plan: initial, revised, checkpoint", required=False, default="initial"),
            ToolParameter("metadata", "object", "Additional metadata about the plan", required=False, default={}),
        ],
        returns="Plan recording result",
        category=ToolCategory.CONTEXT_MANAGEMENT,
    )
    def context_tools__record_plan(
        self,
        plan_content: str,
        plan_type: str = "initial",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Record a plan from DefaultAgent for tracking purposes."""
        try:
            timestamp = datetime.now().isoformat()

            plan_record = {
                "item_id": self.item_id,
                "timestamp": timestamp,
                "plan_type": plan_type,
                "content": plan_content,
                "metadata": metadata or {},
                "recorded_by": "context_manager"
            }

            # Save to context-specific plan file
            plan_file = Path.cwd() / f".safeflow_plans_{self.item_id}.json"

            # Load existing plans or create new list
            existing_plans = []
            if plan_file.exists():
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f:
                        existing_plans = json.load(f)
                except Exception:
                    existing_plans = []

            # Add new plan record
            existing_plans.append(plan_record)

            # Save updated plans
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(existing_plans, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "result": {
                    "recorded": True,
                    "plan_file": str(plan_file),
                    "plan_count": len(existing_plans),
                    "timestamp": timestamp
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to record plan: {e}"}

    @tool_function(
        description="Retrieve recorded plans for the current session.",
        parameters=[
            ToolParameter("plan_type", "string", "Filter by plan type, or 'all' for all plans", required=False, default="all"),
            ToolParameter("limit", "integer", "Maximum number of plans to return", required=False, default=10),
        ],
        returns="List of recorded plans",
        category=ToolCategory.CONTEXT_MANAGEMENT,
    )
    def context_tools__get_recorded_plans(
        self,
        plan_type: str = "all",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Retrieve recorded plans for the current session."""
        try:
            plan_file = Path.cwd() / f".safeflow_plans_{self.item_id}.json"

            if not plan_file.exists():
                return {
                    "success": True,
                    "result": {
                        "plans": [],
                        "count": 0,
                        "message": "No plans recorded yet"
                    }
                }

            with open(plan_file, 'r', encoding='utf-8') as f:
                all_plans = json.load(f)

            # Filter by type if specified
            if plan_type != "all":
                filtered_plans = [p for p in all_plans if p.get("plan_type") == plan_type]
            else:
                filtered_plans = all_plans

            # Apply limit
            limited_plans = filtered_plans[-limit:] if limit > 0 else filtered_plans

            return {
                "success": True,
                "result": {
                    "plans": limited_plans,
                    "count": len(limited_plans),
                    "total_plans": len(all_plans)
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to retrieve plans: {e}"}

    @tool_function(
        description="Record conversation memory or important decisions.",
        parameters=[
            ToolParameter("memory_type", "string", "Type: decision, milestone, error, insight", required=True),
            ToolParameter("content", "string", "Memory content to record", required=True),
            ToolParameter("context", "object", "Additional context information", required=False, default={}),
        ],
        returns="Memory recording result",
        category=ToolCategory.CONTEXT_MANAGEMENT,
    )
    def context_tools__record_memory(
        self,
        memory_type: str,
        content: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Record important conversation memory or decisions."""
        try:
            timestamp = datetime.now().isoformat()

            memory_record = {
                "item_id": self.item_id,
                "timestamp": timestamp,
                "type": memory_type,
                "content": content,
                "context": context or {},
                "recorded_by": "context_manager"
            }

            # Save to context-specific memory file
            memory_file = Path.cwd() / f".safeflow_memory_{self.item_id}.json"

            # Load existing memories or create new list
            existing_memories = []
            if memory_file.exists():
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        existing_memories = json.load(f)
                except Exception:
                    existing_memories = []

            # Add new memory record
            existing_memories.append(memory_record)

            # Keep only recent memories (last 100)
            if len(existing_memories) > 100:
                existing_memories = existing_memories[-100:]

            # Save updated memories
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(existing_memories, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "result": {
                    "recorded": True,
                    "memory_file": str(memory_file),
                    "memory_count": len(existing_memories),
                    "timestamp": timestamp
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to record memory: {e}"}

    @tool_function(
        description="Get session summary including plans and key memories.",
        parameters=[],
        returns="Session summary for coordination with DefaultAgent",
        category=ToolCategory.CONTEXT_MANAGEMENT,
    )
    def context_tools__get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary for coordination."""
        try:
            # Get recent plans
            plans_result = self.context_tools__get_recorded_plans(limit=5)
            recent_plans = plans_result.get("result", {}).get("plans", [])

            # Get recent memories
            memory_file = Path.cwd() / f".safeflow_memory_{self.item_id}.json"
            recent_memories = []

            if memory_file.exists():
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        all_memories = json.load(f)
                        recent_memories = all_memories[-10:]  # Last 10 memories
                except Exception:
                    pass

            return {
                "success": True,
                "result": {
                    "item_id": self.item_id,
                    "recent_plans": recent_plans,
                    "recent_memories": recent_memories,
                    "summary_timestamp": datetime.now().isoformat(),
                    "total_plans": len(recent_plans),
                    "total_memories": len(recent_memories)
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to generate session summary: {e}"}

    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "item_id": self.item_id,
            "functions": len(self.functions) if hasattr(self, 'functions') else 0,
        }