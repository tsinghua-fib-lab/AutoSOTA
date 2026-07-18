import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from rich.console import Console
from tools.abs_tools import ToolRegistry
from tools.file_system import FileSystemTool
from tools.windowed_editor import WindowedEditorTool
from tools.planning_tool import PlanningTool
from tools.base_tools import BaseTools
from tools.env_management import EnvManagementTool
from tools.skills_tools import SkillsTools
from traj import TraceTrack
from agent.context_manager import ContextManagerAgent
from tools.analysis import AnalysisTools

logger = logging.getLogger(name=__name__)
agent_console = Console()

class DefaultAgent:
    """
    This is the code agent.
    """
    def __init__(self, config, item_id: str, work_root: str = None, agent_name: str = "default", addtional_sys_message: str = "") -> None:
        self.config = config
        self.max_turns = self.config.max_turns
        self.api_key = self.config.api_key
        self.api_url = self.config.api_url
        self.max_retry = self.config.max_retry
        self.agent_name = agent_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.model = self.config.model_name
        self.item_id = item_id
        self.work_root = work_root
        self.trace_track = TraceTrack(root_dir=self.config.output_dir, agent_name=agent_name, item_id=item_id)

        # Initialize session state
        self.plan_created = False

        # Test retry limitation to prevent infinite loops
        self.test_attempt_count = 0
        self.max_test_attempts = 5  # Maximum number of test retry attempts
        self.consecutive_test_failures = 0
        self.max_consecutive_failures = 3  # Stop if tests fail 3 times in a row

        # Initialize tool registry with all available tools
        self.tool_registry = ToolRegistry()

        # Register all tools with same item_id
        self.tool_registry.register_tool(FileSystemTool(item_id=item_id))
        self.tool_registry.register_tool(WindowedEditorTool(item_id=item_id))
        self.tool_registry.register_tool(PlanningTool(item_id=item_id))
        self.tool_registry.register_tool(AnalysisTools(item_id=item_id))
        trigger = getattr(getattr(self.config, "attack_config", None), "trigger", "")
        self.tool_registry.register_tool(SkillsTools(item_id=item_id, trigger=trigger))

        # Initialize BaseTools with work_root
        self.base_tools = BaseTools(item_id=item_id, work_root=work_root)
        self.tool_registry.register_tool(self.base_tools)

        # Register EnvManagementTool with work_root provider
        self.tool_registry.register_tool(EnvManagementTool(
            item_id=item_id,
            work_root_provider=lambda: self.work_root
        ))

        # Initialize Context Manager Agent for state and memory management
        self.context_manager = ContextManagerAgent(
            config=config,
            item_id=item_id,
            default_agent_name=agent_name
        )
        self.context_manager.update_work_context(work_root=self.work_root)

        agent_console.print("✅ All Tools registered\n", style="green")
        agent_console.print(self.tool_registry.get_registry_summary())

        # Simplified system message focused on core functionality
        work_root_info = f"Working Directory: {self.work_root or 'Not specified'}"

        self.system_message = f"""You are SafeFlow, an intelligent code analysis and repair agent.

## Current Context
{work_root_info}

## Available Tools
- **file_system**: Find, search, and analyze code files
- **windowed_editor**: Edit files with precise line control
- **env_management**: Run tests, execute commands, check code quality
- **base_tools**: Basic operations (git status, test running, task completion)

## Your Process
1. **Understand**: Read the problem description carefully
2. **Explore**: Use file_system tools to find relevant code
3. **Analyze**: Understand the current behavior and root cause
4. **Fix**: Make targeted changes using windowed_editor
5. **Verify**: Run tests to confirm your fix works
6. **Complete**: Use base_tools__finish_task when done

## Guidelines
- Always use absolute paths: {self.work_root or '/path/to/work'}/filename
- Make minimal, targeted changes
- Test your changes frequently
- Focus on the specific issue described

## Testing Guidelines
- Use env_management__execute_bash to run tests
- Verify both that failing tests now pass AND existing tests still work
- Run `python -m pytest test_path -v` for detailed test output
- **IMPORTANT**: If tests fail repeatedly (3+ times), consider generating a patch instead of retrying
- Focus on creating a working solution rather than achieving 100% test pass rate

## Completion Options
1. **Generate Patch (Recommended for SWE tasks)**: Use `base_tools__generate_patch()` to create a patch file with your changes
2. **Direct Completion**: Use `base_tools__finish_task()` for regular tasks
3. **Final Return**: If you have any answers to return, please put them in the "message" parameter of `base_tools__finish_task()`. Also include the returns (if you have) in the content of your final round responses at the same time.

Start by understanding the problem and exploring the codebase. \n{addtional_sys_message}"""

    def get_current_work_context(self) -> Dict[str, Any]:
        """Get current work context from context manager."""
        if self.context_manager:
            return self.context_manager.get_current_context()

        # Fallback: try to get from base_tools
        for tool_name, tool in self.tool_registry.tools.items():
            if tool_name == "base_tools" and hasattr(tool, 'get_current_work_root'):
                return {
                    "work_root": str(tool.get_current_work_root()),
                    "message": "Context from base_tools (limited info)"
                }

        return {"work_root": None, "message": "No context available"}
    
    def chat_with_retry(self, model, messages,tools, tool_choice="auto",):
        last_err = None
        for i in range(self.max_retry):
            try:
                return self.client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice=tool_choice)
            except Exception as e:
                last_err = e
                time.sleep(2 * i + 1)  # 指数退避
                agent_console.print(f"API error: {e}")
        raise last_err

    def before_tool_call(self, tool_name: str, function_name: str) -> str:
        """Get context information before tool calls."""
        context = self.get_current_work_context()
        context_summary = ""

        if self.context_manager:
            context_summary = self.context_manager.get_context_summary_for_agent()
        else:
            context_summary = f"Work Root: {context.get('work_root', 'Not set')}"

        return context_summary

    def update_context_after_tool_call(self, tool_name: str, function_name: str,
                                     tool_result: Dict[str, Any], parameters: Dict[str, Any]) -> None:
        """Update context after tool calls."""
        if not self.context_manager:
            return

        # Update context based on tool usage
        if tool_name == "base_tools":
            if function_name == "base_tools__set_work_root" and tool_result.get("success"):
                work_root = tool_result.get("result", {}).get("work_root")
                if work_root:
                    self.context_manager.update_work_context(work_root=work_root)

        elif tool_name == "windowed_editor" or tool_name == "file_system":
            # Track file operations
            if "path" in parameters:
                file_path = parameters["path"]
                operation = function_name.split("__")[-1] if "__" in function_name else function_name
                self.context_manager.track_active_file(file_path, operation)

    def get_enhanced_system_message(self) -> str:
        """Get system message enhanced with current context."""
        base_message = self.system_message

        if self.context_manager:
            context_work_root = self.context_manager.current_state["work_root"]
            enhanced_message = f"{base_message}\n\nCURRENT WORK CONTEXT:\n Now, you are working at dirs: {context_work_root}\n\n"


            return enhanced_message

        return base_message

    def _route_function(self, openai_function_name: str) -> Tuple[str, str]:
        """
        Map OpenAI tool-call function name -> (tool_name, function_name)

        Your registry is keyed by tool_name then function_name.
        OpenAI tool-call uses only function name in schema; so default to 'file_system'.
        If you later choose to emit names like 'file_system.write_file', this supports it.
        """
        fn = openai_function_name
        for tool_name, tool in self.tool_registry.tools.items():
            # tool.functions keys are callable names on that tool
            if fn in getattr(tool, "functions", {}):
                return tool_name, fn
        raise ValueError(f"No tool found for function {fn}")

    def _enhance_user_prompt_with_planning(self, user_prompt: str) -> str:
        """
        Enhance user prompt with planning requirements if needed.
        """
        if self.plan_created:
            return user_prompt

        self.plan_created = True
        return (
            f"USER TASK: {user_prompt}\n\n"
            "PLANNING REQUIREMENT: You must create a plan before starting work.\n"
            "1. First call: planning_tool__create_plan() to break down the task into steps.\n"
            "2. Then proceed with executing the plan step by step.\n\n"
            "Remember: You MUST create a plan before starting work on the task."
        )

    def _check_plan_creation(self, function_name: str, tool_result: Dict[str, Any]) -> None:
        """
        Monitor tool calls to detect plan creation.
        """
        if function_name == "planning_tool__create_plan" and tool_result.get("success"):
            if self.context_manager:
                # Extract plan content from result
                plan_data = tool_result.get("result", {})
                plan_content = plan_data.get("plan_content", "Plan created successfully")

                # Record the plan in context manager
                self.context_manager.record_plan(
                    plan_content=plan_content,
                    plan_type="initial",
                    metadata={
                        "function_name": function_name,
                        "result": plan_data
                    }
                )
    def run(self, user_prompt: str):
        trace_track = self.trace_track

        # Record task start with context manager if available
        if self.context_manager:
            self.context_manager.record_memory(
                "task_start",
                f"DefaultAgent started task: {user_prompt}"
            )
        tools_schema = [
            {"type": "function", "function": f}
            for f in self.tool_registry.to_openai_functions(enabled_only=True)
        ]

        messages: List[Dict[str, Any]] = []

        def push(msg: Dict[str, Any]):
            """同时写入：给模型的 messages + 轨迹 trace(json, 带 time_stamp) + context_manager memory"""
            messages.append(msg)
            trace_track.add_message(msg)

            # Also record in context manager if available
            if self.context_manager:
                self.context_manager.add_default_agent_message(msg)

        # 初始化消息 - 使用增强的系统消息
        enhanced_system_message = self.get_enhanced_system_message()
        push({"role": "system", "content": enhanced_system_message})

        # 检查并引导规划
        enhanced_prompt = self._enhance_user_prompt_with_planning(user_prompt)
        push({"role": "user", "content": enhanced_prompt})

        # 保存工具schema到outputs目录便于调试和分析
        trace_track.save_json(f"{self.agent_name}_tools_schema.json", tools_schema)

        for turn in range(1, self.max_turns + 1):

            # Check if plan reminder is needed (context manager)
            if self.context_manager:
                reminder_data = self.context_manager.check_plan_reminder_needed()
                if reminder_data:
                    # Add reminder message to conversation
                    reminder_msg = {"role": "system", "content": reminder_data["message"]}
                    push(reminder_msg)

                # Add working directory reminder each turn
                if self.context_manager and self.context_manager.should_remind_work_dir() and turn %10 == 0:
                    work_dir_reminder = self.context_manager.get_working_directory_reminder()
                    work_dir_msg = {"role": "system", "content": work_dir_reminder}
                    push(work_dir_msg)

            # Use context_manager's memory management to get appropriately sized messages
            # This ensures we don't exceed token limits while preserving important context
            actual_messages_for_llm = messages
            if self.context_manager:
                try:
                    # Get summarized messages if needed, otherwise original messages
                    summarized_messages = self.context_manager.get_summarized_messages()
                    if summarized_messages != self.context_manager.default_agent_messages:
                        # Context manager applied summarization, use the summarized version
                        actual_messages_for_llm = summarized_messages
                        agent_console.print(f"🧠 Using summarized messages: {len(summarized_messages)} vs {len(messages)}", style="blue")
                except Exception as e:
                    logger.warning(f"Failed to get summarized messages, using original: {e}")
                    actual_messages_for_llm = messages


            resp = self.chat_with_retry(
                model=self.model,
                messages=actual_messages_for_llm,
                tools=tools_schema,
                tool_choice="auto",
            )

            messages = actual_messages_for_llm

            choice = resp.choices[0]
            msg = choice.message
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None) or []
            # assistant message（带 tool_calls 时 content 可能为 None，这没问题）
            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": content}

            # 美化输出显示
            if content:
                agent_console.print(f"[bold green]Agent Turn {turn}[/bold green]: {content}")
            elif tool_calls:
                tool_names = [tc.function.name for tc in tool_calls]
                agent_console.print(f"[bold green]Agent Turn {turn}[/bold green]: [cyan]Using tools[/cyan]: {', '.join(tool_names)}")
            else:
                agent_console.print(f"[bold green]Agent Turn {turn}[/bold green]: [yellow]No response content[/yellow]")

            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            push(assistant_entry)

            # 1) 没有 tool_calls，进入下一轮，因为结束也得tool_calss
            if not tool_calls:
                nt_calls = {"role": "user", "content": "What do not you call the tools? If you think task is finished, just calling `base_tools__finish_task()` to finish the task. If you have any answers to return, please put them in the 'message' parameter of `base_tools__finish_task()`. Also include the returns (if you have) in the content of your final round responses at the same time."}
                push(msg=nt_calls)
                continue

            # 2) 有 tool_calls：逐个执行工具
            for tc in tool_calls:
                fn_name = tc.function.name
                args_raw = tc.function.arguments or "{}"
                try:
                    tool_name, function_name = self._route_function(fn_name)
                except Exception as e:
                    tool_result = {"success": False, "error": f"Bad tool names: {e}", "raw": fn_name}

                # 解析参数（失败也要回填 tool 消息）
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object/dict")
                except Exception as e:
                    tool_result = {"success": False, "error": f"Bad tool arguments: {e}", "raw": args_raw}
                else:
                    tool_result = self.tool_registry.call_function(
                        tool_name=tool_name,
                        function_name=function_name,
                        parameters=args,
                    )

                # tool message：为了贴合你给的示例，加上 name 字段（可选但推荐）
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,  # 例如 "write_file" / "finish_task"
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                push(tool_msg)

                # Monitor plan creation
                self._check_plan_creation(function_name, tool_result)

                # Monitor test attempts to prevent infinite loops
                self._monitor_test_attempts(function_name, tool_result)

                # Update context after tool call
                self.update_context_after_tool_call(tool_name, function_name, tool_result, args)

                # 3) 如果调用 finish_task：提前退出（推荐只用它作为"完成信号"）
                if function_name == "base_tools__finish_task":
                    # Record task completion with context manager if available
                    if self.context_manager:
                        self.context_manager.record_memory(
                            "task_completed",
                            args["message"] or content
                        )
                    return {"success": True, "final":args["message"] or content, "messages": messages}

        return {
            "success": False,
            "error": f"Reached max_turns={self.max_turns} without finish_task or final response.",
            "messages": messages,
        }

    def _monitor_test_attempts(self, function_name: str, tool_result: Dict[str, Any]) -> None:
        """
        Monitor test attempts to prevent infinite testing loops.
        """
        # Check if this is a test-related function
        test_functions = [
            "env_management__execute_bash",  # Running pytest directly
        ]

        # Check if the bash command looks like a test command
        is_test_command = False
        if function_name == "env_management__execute_bash":
            # Check if result contains test-related keywords
            result_str = str(tool_result).lower()
            if any(keyword in result_str for keyword in ["pytest"]):
                is_test_command = True
        elif function_name in test_functions:
            is_test_command = True

        if is_test_command:
            self.test_attempt_count += 1

            # Check if test failed
            test_success = tool_result.get("success", False)
            if not test_success:
                self.consecutive_test_failures += 1
            else:
                self.consecutive_test_failures = 0  # Reset on success

            # Log warning if approaching limits
            if self.test_attempt_count >= self.max_test_attempts - 1:
                agent_console.print(
                    f"⚠️ [yellow]Warning[/yellow]: Test attempt {self.test_attempt_count}/{self.max_test_attempts}. "
                    "Consider generating a patch if tests keep failing.",
                    style="yellow"
                )
            elif self.consecutive_test_failures >= self.max_consecutive_failures - 1:
                agent_console.print(
                    f"⚠️ [yellow]Warning[/yellow]: {self.consecutive_test_failures} consecutive test failures. "
                    "Consider alternative approach or generate patch.",
                    style="yellow"
                )

            # Record in context manager if available
            if self.context_manager:
                self.context_manager.record_memory(
                    "test_attempt",
                    f"Test attempt {self.test_attempt_count}: {function_name} - {'Success' if test_success else 'Failed'}"
                )
