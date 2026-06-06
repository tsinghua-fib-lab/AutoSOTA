SHORT_TERM_PROMPT_TEMPLATE = """ 
TASK: Infer the user's persona based on their ratings of {item_type} items.   
Instructions: 
Below is a list of {item_type}s that the user has rated. 
Each rating ranges from 1 to 5:  
{actual}   

Based on these, generate a user persona(at least 190 words) without mentioning item names or rating scores. 
User Persona:  
""".strip()

HIER_MERGE_PROMPT_TEMPLATE = """ 
TASK: Update the long-term persona by integrating it with the newly generated short-term persona.  

Instructions:  
Below is the existing long-term persona based on prior behaviors:  
{long_term_persona}  

Below is the newly generated short-term persona based on recent behaviors:  
{short_term_persona}  

Integrate the short-term persona into the long-term persona to capture both historical stability and recent dynamics.  
The updated persona should reflect both long-term preferences and recent changes without losing consistency.  
Updated Long-Term Persona:  
""".strip()
