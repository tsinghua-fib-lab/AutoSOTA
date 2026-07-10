#!/bin/bash
# CODE-02 Part 2: Update conversation.py to pass criteria_objs to assistant
set -e
cd /repo

cp /repo/discoverllm/simulate/conversation.py /repo/patches/conversation.py.bak

python3 << "PYEOF"
FILE = "/repo/discoverllm/simulate/conversation.py"
with open(FILE) as f:
    content = f.read()

# In run_best_of_1_conversation, modify the assistant_sim call
# Find: assistant_response = assistant_sim(current_message)
# Replace with: assistant_response = assistant_sim(current_message, user_sim.criteria_objs)
old_call = "assistant_response = assistant_sim(current_message)"
new_call = "assistant_response = assistant_sim(current_message, user_sim.criteria_objs)"

if old_call in content:
    content = content.replace(old_call, new_call)
    with open(FILE, "w") as f:
        f.write(content)
    print("CODE-02 conv: assistant_sim now receives criteria_objs")
else:
    print("WARNING: could not find assistant_sim call in conversation.py")
PYEOF
