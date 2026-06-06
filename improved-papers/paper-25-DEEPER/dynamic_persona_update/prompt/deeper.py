DEEPER_PROMPT_TEMPLATE = """
TASK: Refine the old user persona based on differences between predicted and actual ratings of {item_type} items.

Old Persona (Inferred from Past Behavior): 
{old_persona}

Prediction(based on the old persona) vs. Actual Ratings:
{predict_and_actual}

Reflect on these differences and generate a refined user persona without mentioning item names or rating scores.
Refined User Persona:
""".strip()
