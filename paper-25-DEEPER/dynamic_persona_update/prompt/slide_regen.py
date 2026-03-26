SLIDE_REGEN_PROMPT_TEMPLATE = """ 
TASK: Infer the user's persona based on their ratings of {item_type} items.   
Instructions: 
Below is a list of {item_type}s that the user has rated. 
Each rating ranges from 1 to 5:  
{actual}   

Based on these, generate a user persona(at least 190 words) without mentioning item names or rating scores. 
User Persona:  
""".strip()