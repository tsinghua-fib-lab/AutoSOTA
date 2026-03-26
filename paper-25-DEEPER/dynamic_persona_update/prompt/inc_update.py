INC_UPDATE_PROMPT_TEMPLATE = """ 
TASK: Integrate the user's most recent ratings of {item_type} items into their existing persona to generate an updated persona.   

Instructions:  
Below is the existing persona based on prior behaviors:  
{old_persona}  

Below is a list of recent {item_type}s that the user has rated.  
Each rating ranges from 1 to 5:  
{inc_actual}  

Based on these, integrate the new features from the recent ratings into the existing persona.  
Updated Persona:  
""".strip()
