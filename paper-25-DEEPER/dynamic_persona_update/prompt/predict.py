
########### predict ############
BEHAVIOR_PREDICT_PROMPT_TEMPLATE = """
TASK: Role-play the given persona and predict what score (out of 5) you would give to the following {item_type} list.

Instructions: Based on the persona: {persona}, predict ratings for each item in the list below.

{items}

Output format:
```json
[
    {{"item_name": "...", "predict_rating": ...}},
    {{"item_name": "...", "predict_rating": ...}},
    ...
]
```
""".strip()