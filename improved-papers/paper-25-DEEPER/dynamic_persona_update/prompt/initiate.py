PERSONA_INITIATE_PROMPT_TEMPLATE = """
TASK: Infer the user's persona based on their previous behaviors.

Instructions: Below is a list of {item_type}s that the user has rated. Each rating ranges from 1 to 5:

{prev_user_behaviours_with_items}

Using these ratings, please deduce the user's likely persona. 

### Answer Format:
```json
{{
   "persona": <result>
}}
- <result> should be a description of at least  **200 words**.
- Do not quote {item_type} titles.
""".strip()