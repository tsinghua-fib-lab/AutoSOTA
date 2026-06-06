"""
Snapshot Descriptor prompt code for SAVVY pipeline stage1 - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

def get_sd_prompt_v2(uploaded_obj, question):
    """
    Generates a simple prompt for tracking specific objects audio-visually
    throughout a video and outputting structured JSON data.

    Args:
        uploaded_obj (str): Path or name of the uploaded video file.
        question (str): User's question about objects and sound in the video.

    Returns:
        str: The generated prompt string.
    """
    return f"""
        Analyze the video at `{uploaded_obj}` based on the question: "{question}".
        Identify the **Sounding Object**, the **Reference Object**, and the **Facing Object** (stand by the **Reference Object** and face the **Facing Object**).
        Identify the **start_time** and **end_time** of the event mentioned in the question.
        Determine the mode:
            - If I'm in the **camera's view** (egocentric), set `mode` to `egocentric`. 
            - If I'm in a **different perspective** rather than the camera's view (allocentric), set `mode` to `allocentric`. 

        Perform audio-visual tracking for these objects throughout the *entire duration* of the video.

        **Tracking Data:**
        -   For each object, provide its estimated position over time.
        -   Record positions at key moments across the *full video timeline* when the object is clearly visible in the frame.
        -   Estimate distance in meters from the camera to the object center.
        -   Estimate direction in degrees (-90 left to 90 right, 0 forward) from the camera.

        **Output:**
        Your complete and sole output must be a single JSON object with the following structure:
        

        ```json
        {{
            "event": "Brief description of the event from the question",
            "start_time": "minutes:seconds", 
            "end_time": "minutes:seconds",
            "mode": egocentric/allocentric,
            "sounding_object": {{ 
                "description": "A detailed description of the sounding object. Include physical characteristics like type, color, material, and approximate size/shape.",
                "is_static": true/false, // Set to true if the object is generally non-moving (like furniture, walls) and false if it typically moves location (like a person, animal, vehicle).
                "key_frames": {{ //*entire video* key visible frames
                    "minutes:seconds": {{"distance": "meters", "direction": "degrees"}}
                    // ...
                }}
            }},
            "reference_object": {{ // Stand by Reference Object or camera
                "object_name": "Name",
                "description": "Description",
                "key_frames": {{//*entire video* key visible frames
                    "minutes:seconds": {{"distance": "meters", "direction": "degrees"}}
                }}
                
            }},
            "facing_object": {{ // Facing the facing_object, empty for camera
                "object_name": "Name",
                "description": "Description",
                "key_frames": {{ // *entire video* key visible frames
                    "minutes:seconds": {{"distance": "meters", "direction": "degrees"}}
                }}
                
            }},
            "prediction": // answer of the question, a single number or letter
        }}
        ```
        """