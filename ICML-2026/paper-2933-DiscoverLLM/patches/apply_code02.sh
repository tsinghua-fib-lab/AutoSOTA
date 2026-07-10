#!/bin/bash
# CODE-02: Intent State Sidebar as Supplementary Context
# Adds structured intent state to assistant system prompt prefix
set -e
cd /repo

# Backup
cp /repo/discoverllm/pipeline/assistant_simulator.py /repo/patches/assistant_simulator.py.bak

python3 << "PYEOF"
import re

FILE = "/repo/discoverllm/pipeline/assistant_simulator.py"
with open(FILE) as f:
    content = f.read()

# Add import for get_goal_status_as_str
old_import = "from discoverllm.core.generate import generate_and_process, generate_chat"
new_import = """from discoverllm.core.generate import generate_and_process, generate_chat
from discoverllm.core.tasks.simulate_user_response import get_goal_status_as_str"""

content = content.replace(old_import, new_import)

# Modify __call__ to accept optional criteria_objs for sidebar
old_def = """    def __call__(self, user_response: str):
        self.chat_history.append({"role": "user", "content": user_response})"""

new_def = """    def __call__(self, user_response: str, criteria_objs=None):
        self.chat_history.append({"role": "user", "content": user_response})
        
        # CODE-02: Intent State Sidebar
        # Prepend structured intent state as a supplementary system prefix
        if criteria_objs is not None:
            try:
                state_str = get_goal_status_as_str(criteria_objs)
                if state_str:
                    sidebar = (
                        "[INTENT STATE — discovered and pending intents from prior turns]\n"
                        + state_str
                        + "\n[/INTENT STATE]\n\n"
                    )
                    # Prepend sidebar to the last user message for context
                    self.chat_history[-1]["content"] = sidebar + self.chat_history[-1]["content"]
            except Exception:
                pass  # Graceful degradation if state extraction fails"""

content = content.replace(old_def, new_def)

with open(FILE, "w") as f:
    f.write(content)
print("CODE-02 applied: intent state sidebar added to AssistantSimulator.__call__")
PYEOF
