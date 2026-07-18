"""
Planning Tool for SafeFlow - Plan with Files

Provides atomic planning operations for agents to:
- Create and manage execution plans
- Track progress against plans
- Update plans based on feedback
- Maintain planning context across sessions

Design principles:
- Atomic operations: each tool call performs one well-defined action.
- Stateless tools: all state is persisted in plan files; the tool does not hold session state.
- Absolute-path only: all file paths must be absolute; tool validates this.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from tools.abs_tools import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class PlanningTool(Tool):
    """
    Atomic Planning Tool for SafeFlow.

    Manages task planning and execution tracking through file-based persistence.
    Supports iterative plan refinement and progress monitoring.
    """

    def __init__(
        self,
        item_id: str,
        name: str = "planning_tool",
        description: str = "Task planning and progress management",
    ):
        super().__init__(
            name=name,
            description=description,
            category=ToolCategory.PLAN_TOOLS
        )

        self.item_id = item_id

    def _validate_absolute_path(self, path: str) -> Path:
        """Validate absolute path and return resolved Path object."""
        p = Path(path)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute, got: {path}")
        return p.resolve()

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _load_plan(self, plan_file: Path) -> Optional[Dict[str, Any]]:
        """Load existing plan from file."""
        if not plan_file.exists():
            return None

        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load plan from {plan_file}: {e}")
            return None

    def _save_plan(self, plan_file: Path, plan_data: Dict[str, Any]) -> None:
        """
        Save plan to file (atomic write).

        To preserve atomicity and reduce corruption risk in case of interruptions,
        we write to a temporary file in the same directory and then os.replace().
        """
        plan_file.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = plan_file.with_suffix(plan_file.suffix + ".tmp")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, plan_file)

    @tool_function(
        description=(
            "Create a new execution plan for the current task.\n"
            "- Atomic operation: creates a new plan file.\n"
            "- Stateless: plan is persisted to the plan_file.\n"
            "- Requires absolute plan_file path."
        ),
        parameters=[
            ToolParameter("plan_file", "string", "Absolute path to plan file", required=True),
            ToolParameter("task_description", "string", "High-level description of the task", required=True),
            ToolParameter("overall_goal", "string", "The main objective to accomplish", required=True),
            ToolParameter("steps", "array", "List of planned execution steps", required=True),
            ToolParameter("context_info", "string", "Additional context about the task", required=False, default=""),
        ],
        returns="Plan creation result",
        category=ToolCategory.PLAN_TOOLS,
    )
    def planning_tool__create_plan(
        self,
        plan_file: str,
        task_description: str,
        overall_goal: str,
        steps: List[str],
        context_info: str = ""
    ) -> Dict[str, Any]:
        """Create a new execution plan and save to file."""
        try:
            plan_path = self._validate_absolute_path(plan_file)

            existing_plan = self._load_plan(plan_path)
            if existing_plan:
                return {
                    "success": False,
                    "error": f"Plan already exists at {plan_path}. Use update_plan to modify or view_plan to see current plan."
                }

            plan_data = {
                "metadata": {
                    "created_at": self._get_timestamp(),
                    "updated_at": self._get_timestamp(),
                    "item_id": self.item_id,
                    "version": 1,
                },
                "task": {
                    "description": task_description,
                    "overall_goal": overall_goal,
                    "context_info": context_info,
                },
                "execution": {
                    "steps": [
                        {
                            "id": i + 1,
                            "description": step,
                            "status": "pending",  # pending, in_progress, completed, failed
                            "notes": "",
                            "completed_at": None,
                        }
                        for i, step in enumerate(steps)
                    ],
                    "current_step": 1,
                    "overall_status": "active",  # active, paused, completed, failed
                },
                "history": [
                    {
                        "timestamp": self._get_timestamp(),
                        "action": "plan_created",
                        "description": f"Initial plan created with {len(steps)} steps",
                    }
                ]
            }

            self._save_plan(plan_path, plan_data)

            result = {
                "plan_file": str(plan_path),
                "steps_count": len(steps),
                "current_step": 1,
                "plan_content": "\n".join([f"{i + 1}. {step}" for i, step in enumerate(steps)]),
                "plan_summary": {
                    "task": task_description,
                    "goal": overall_goal,
                    "total_steps": len(steps),
                    "first_step": steps[0] if steps else None,
                }
            }

            logger.info(f"Created plan with {len(steps)} steps: {plan_path}")
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Failed to create plan: {e}"}

    @tool_function(
        description=(
            "View the current execution plan and progress.\n"
            "- Atomic operation: reads plan file and returns a snapshot.\n"
            "- Stateless: does not modify plan.\n"
            "- Requires absolute plan_file path."
        ),
        parameters=[
            ToolParameter("plan_file", "string", "Absolute path to plan file", required=True),
            ToolParameter("show_completed", "boolean", "Include completed steps in output", required=False, default=True),
        ],
        returns="Current plan and progress status",
        category=ToolCategory.PLAN_TOOLS,
    )
    def planning_tool__view_plan(
        self,
        plan_file: str,
        show_completed: bool = True
    ) -> Dict[str, Any]:
        """View current plan and progress status."""
        try:
            plan_path = self._validate_absolute_path(plan_file)

            plan_data = self._load_plan(plan_path)
            if not plan_data:
                return {"success": False, "error": f"No plan found at {plan_path}"}

            task = plan_data.get("task", {})
            execution = plan_data.get("execution", {})
            steps = execution.get("steps", [])

            if show_completed:
                display_steps = steps
            else:
                display_steps = [step for step in steps if step.get("status") != "completed"]

            completed_count = len([s for s in steps if s.get("status") == "completed"])
            total_count = len(steps)
            progress_percentage = (completed_count / total_count * 100) if total_count > 0 else 0

            current_step_id = execution.get("current_step", 1)
            current_step = next((s for s in steps if s.get("id") == current_step_id), None)

            result = {
                "plan_file": str(plan_path),
                "task_info": {
                    "description": task.get("description", ""),
                    "overall_goal": task.get("overall_goal", ""),
                    "context_info": task.get("context_info", ""),
                },
                "progress": {
                    "completed_steps": completed_count,
                    "total_steps": total_count,
                    "percentage": round(progress_percentage, 1),
                    "current_step_id": current_step_id,
                    "overall_status": execution.get("overall_status", "active"),
                },
                "current_step": current_step,
                "steps": display_steps,
                "metadata": plan_data.get("metadata", {}),
            }

            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Failed to view plan: {e}"}

    @tool_function(
        description=(
            "Update plan progress or modify plan content.\n"
            "Supported actions:\n"
            "- complete_step: mark a step completed\n"
            "- fail_step: mark a step failed\n"
            "- add_step: append a new step\n"
            "- update_step: update an existing step description and/or notes\n"
            "- set_current_step: set current_step (also marks pending -> in_progress)\n"
            "\n"
            "Atomic: reads, modifies, writes plan file in one call.\n"
            "Stateless: all state in plan_file.\n"
            "Absolute path required."
        ),
        parameters=[
            ToolParameter("plan_file", "string", "Absolute path to plan file", required=True),
            ToolParameter(
                "action",
                "string",
                "Action to perform: complete_step, fail_step, add_step, update_step, set_current_step",
                required=True
            ),
            ToolParameter("step_id", "integer", "ID of step to operate on", required=False),
            ToolParameter("step_description", "string", "New or updated step description", required=False),
            ToolParameter("notes", "string", "Notes about the step or update (empty string will clear step notes for update_step/complete_step/fail_step/add_step)", required=False, default=""),
        ],
        returns="Plan update result",
        category=ToolCategory.PLAN_TOOLS,
    )
    def planning_tool__update_plan(
        self,
        plan_file: str,
        action: str,
        step_id: Optional[int] = None,
        step_description: Optional[str] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Update plan progress or modify plan content.

        Notes semantics:
        - For update_step: notes is treated as the new step notes. If notes == "" it clears step notes.
        - For complete_step/fail_step/add_step: if notes is provided (including ""), it sets step["notes"]=notes.
          This is to avoid ambiguity while keeping the same signature. (Previously, empty string meant "ignore".)
        """
        try:
            plan_path = self._validate_absolute_path(plan_file)

            plan_data = self._load_plan(plan_path)
            if not plan_data:
                return {"success": False, "error": f"No plan found at {plan_path}"}

            execution = plan_data.setdefault("execution", {})
            steps = execution.setdefault("steps", [])
            history = plan_data.setdefault("history", [])

            timestamp = self._get_timestamp()
            allowed_status = {"pending", "in_progress", "completed", "failed"}

            normalized_steps: List[int] = []

            def _get_step(sid: int) -> Optional[Dict[str, Any]]:
                return next((s for s in steps if s.get("id") == sid), None)

            def _append_history(act: str, desc: str, extra: Optional[Dict[str, Any]] = None):
                rec = {"timestamp": timestamp, "action": act, "description": desc}
                # Keep existing signature: notes is always available; record it for audit.
                rec["notes"] = notes
                if extra:
                    rec.update(extra)
                history.append(rec)

            def _normalize_statuses():
                """Fix invalid statuses to pending and record it."""
                for s in steps:
                    st = s.get("status", "pending")
                    if st not in allowed_status:
                        old = st
                        s["status"] = "pending"
                        normalized_steps.append(s.get("id"))
                        _append_history(
                            "status_normalized",
                            f"Normalized invalid status for step {s.get('id')}: {old!r} -> 'pending'",
                        )

            def _sorted_steps():
                """Return steps sorted by integer id (robust if list order got changed)."""
                def _id(s: Dict[str, Any]) -> int:
                    try:
                        return int(s.get("id", 0))
                    except Exception:
                        return 0
                return sorted(steps, key=_id)

            def _fix_current_step():
                """
                Ensure current_step points to an actionable step.
                Prefer in_progress, then pending. If none, keep current_step as-is.
                """
                current_id = execution.get("current_step", 1)
                current = _get_step(current_id)
                if current is not None and current.get("status") not in {"completed", "failed"}:
                    return

                # Prefer in_progress, then pending (sorted by id)
                for st in ("in_progress", "pending"):
                    candidate = next((s for s in _sorted_steps() if s.get("status") == st), None)
                    if candidate is not None:
                        execution["current_step"] = candidate.get("id")
                        return

            def _update_overall_status():
                if any(s.get("status") == "failed" for s in steps):
                    execution["overall_status"] = "failed"
                elif steps and all(s.get("status") == "completed" for s in steps):
                    execution["overall_status"] = "completed"
                else:
                    execution["overall_status"] = "active"

            def _advance_from_step_id(cur_id: int):
                """
                Advance current_step to the next actionable step (by id order),
                preferring a step with id > cur_id, else the first actionable step.
                """
                ordered = _sorted_steps()
                actionable = [s for s in ordered if s.get("status") in {"in_progress", "pending"}]

                # First try: next step after cur_id
                for s in actionable:
                    try:
                        if int(s.get("id")) > int(cur_id):
                            execution["current_step"] = s.get("id")
                            return
                    except Exception:
                        continue

                # Fallback: first actionable
                if actionable:
                    execution["current_step"] = actionable[0].get("id")
                else:
                    # no actionable steps; keep current_step (caller can infer from overall_status)
                    pass

            _normalize_statuses()

            if action == "complete_step":
                if step_id is None:
                    return {"success": False, "error": "step_id required for complete_step action"}

                step = _get_step(step_id)
                if not step:
                    return {"success": False, "error": f"Step {step_id} not found"}

                step["status"] = "completed"
                step["completed_at"] = timestamp
                # Set notes even if empty string (clears if caller wants)
                step["notes"] = notes

                if execution.get("current_step") == step_id:
                    _advance_from_step_id(step_id)

                _append_history("step_completed", f"Completed step {step_id}: {step.get('description', '')}")

            elif action == "fail_step":
                if step_id is None:
                    return {"success": False, "error": "step_id required for fail_step action"}

                step = _get_step(step_id)
                if not step:
                    return {"success": False, "error": f"Step {step_id} not found"}

                step["status"] = "failed"
                step["completed_at"] = timestamp
                step["notes"] = notes

                # If failing the current step, advance to next actionable step
                if execution.get("current_step") == step_id:
                    _advance_from_step_id(step_id)

                _append_history("step_failed", f"Failed step {step_id}: {step.get('description', '')}")

            elif action == "add_step":
                if step_description is None or step_description == "":
                    return {"success": False, "error": "step_description required for add_step action"}

                new_id = max([int(s.get("id", 0)) for s in steps] or [0]) + 1
                new_step = {
                    "id": new_id,
                    "description": step_description,
                    "status": "pending",
                    "notes": notes,  # allow empty notes
                    "completed_at": None,
                }
                steps.append(new_step)

                _append_history("step_added", f"Added new step {new_id}: {step_description}")

                _fix_current_step()

            elif action == "set_current_step":
                if step_id is None:
                    return {"success": False, "error": "step_id required for set_current_step action"}

                step = _get_step(step_id)
                if not step:
                    return {"success": False, "error": f"Step {step_id} not found"}

                execution["current_step"] = step_id

                # Make "current" meaningful: pending -> in_progress
                if step.get("status") == "pending":
                    step["status"] = "in_progress"

                _append_history("current_step_changed", f"Changed current step to {step_id}: {step.get('description', '')}")

            elif action == "update_step":
                if step_id is None:
                    return {"success": False, "error": "step_id required for update_step action"}

                step = _get_step(step_id)
                if not step:
                    return {"success": False, "error": f"Step {step_id} not found"}

                old_desc = step.get("description", "")
                old_notes = step.get("notes", "")

                # Allow clearing description only if explicitly set to "" (we accept it)
                if step_description is not None:
                    step["description"] = step_description

                # Allow clearing notes by passing empty string (notes param default is "", so this is consistent)
                step["notes"] = notes

                _append_history(
                    "step_updated",
                    f"Updated step {step_id}",
                    extra={
                        "old_description": old_desc,
                        "new_description": step.get("description", ""),
                        "old_notes": old_notes,
                        "new_notes": step.get("notes", ""),
                    }
                )

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

            _fix_current_step()
            _update_overall_status()

            plan_data.setdefault("metadata", {})
            plan_data["metadata"]["updated_at"] = timestamp
            plan_data["metadata"]["version"] = plan_data.get("metadata", {}).get("version", 0) + 1

            self._save_plan(plan_path, plan_data)

            # Keep output shape stable; only add optional extra fields.
            result = {
                "action_performed": action,
                "step_id": step_id,
                "plan_updated": True,
                "current_step": execution.get("current_step"),
                "overall_status": execution.get("overall_status"),
                "total_steps": len(steps),
            }
            if normalized_steps:
                result["normalized_steps"] = normalized_steps

            logger.info(f"Updated plan: {action} for step {step_id}")
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Failed to update plan: {e}"}

    @tool_function(
        description=(
            "Create a new plan version based on human feedback or task changes.\n"
            "- Atomic operation: revises the plan file in one call.\n"
            "- Stateless: all state persisted in plan_file.\n"
            "- Requires absolute plan_file path."
        ),
        parameters=[
            ToolParameter("plan_file", "string", "Absolute path to plan file", required=True),
            ToolParameter("feedback", "string", "Human feedback or new requirements", required=True),
            ToolParameter("new_steps", "array", "Updated list of execution steps", required=False),
            ToolParameter("keep_completed", "boolean", "Keep already completed steps", required=False, default=True),
        ],
        returns="Plan revision result",
        category=ToolCategory.PLAN_TOOLS,
    )
    def planning_tool__revise_plan(
        self,
        plan_file: str,
        feedback: str,
        new_steps: Optional[List[str]] = None,
        keep_completed: bool = True
    ) -> Dict[str, Any]:
        """Revise plan based on feedback or changed requirements."""
        try:
            plan_path = self._validate_absolute_path(plan_file)

            plan_data = self._load_plan(plan_path)
            if not plan_data:
                return {"success": False, "error": f"No plan found at {plan_path}"}

            timestamp = self._get_timestamp()

            current_steps = plan_data.get("execution", {}).get("steps", [])

            if keep_completed:
                completed_steps = [s for s in current_steps if s.get("status") == "completed"]
                dropped_steps = [s for s in current_steps if s.get("status") != "completed"]
                max_completed_id = max([s.get("id", 0) for s in completed_steps], default=0)
            else:
                completed_steps = []
                dropped_steps = current_steps
                max_completed_id = 0

            if new_steps:
                new_steps_list = []
                new_steps_list.extend(completed_steps)

                for i, step_desc in enumerate(new_steps):
                    new_steps_list.append({
                        "id": max_completed_id + i + 1,
                        "description": step_desc,
                        "status": "pending",
                        "notes": "",
                        "completed_at": None,
                    })

                plan_data["execution"]["steps"] = new_steps_list

                next_pending = next((s for s in new_steps_list if s.get("status") == "pending"), None)
                plan_data["execution"]["current_step"] = next_pending.get("id") if next_pending else max_completed_id + 1

            plan_data["metadata"]["updated_at"] = timestamp
            plan_data["metadata"]["version"] = plan_data.get("metadata", {}).get("version", 0) + 1

            history = plan_data.get("history", [])
            history.append({
                "timestamp": timestamp,
                "action": "plan_revised",
                "description": f"Plan revised based on feedback. New steps: {len(new_steps) if new_steps else 'no change'}",
                "feedback": feedback,
                "dropped_steps": [
                    {"id": s.get("id"), "description": s.get("description"), "status": s.get("status")}
                    for s in dropped_steps
                ] if dropped_steps else []
            })

            self._save_plan(plan_path, plan_data)

            result = {
                "plan_revised": True,
                "feedback_incorporated": feedback,
                "completed_steps_kept": len(completed_steps) if keep_completed else 0,
                "new_steps_added": len(new_steps) if new_steps else 0,
                "total_steps": len(plan_data["execution"]["steps"]),
                "current_step": plan_data["execution"]["current_step"],
            }

            logger.info(f"Revised plan based on feedback: {plan_path}")
            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": f"Failed to revise plan: {e}"}

    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "item_id": self.item_id,
            "functions": len(self.functions) if hasattr(self, 'functions') else 0,
        }