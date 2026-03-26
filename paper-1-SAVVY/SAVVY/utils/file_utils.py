"""
File utils code uploading files to Gemini for SAVVY pipeline stage1 - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import pickle
import time


def load_uploaded_files(filepath="uploaded_gemini_files.pkl"):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_uploaded_files(files, filepath="uploaded_gemini_files.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(files, f)



def check_and_reupload(video_path, uploaded_files, genai):
    """
    Checks the validity of an uploaded object using the genai API, re-uploads if invalid, and updates the file list.

    Args:
        video_path (str): The path to the video file.
        uploaded_files (dict): A dictionary storing uploaded file objects, keyed by video path.
        genai: An object with `upload_file` and `get_file_info` methods.

    Returns:
        The uploaded file object (either existing or re-uploaded).
    """
    if video_path in uploaded_files:
        uploaded_obj = uploaded_files[video_path]
        # Check validity using genai API
        if not is_uploaded_obj_valid(uploaded_obj, genai):
            # print(f"Uploaded object for {video_path} is invalid. Re-uploading...")
            del uploaded_files[video_path]
            uploaded_obj = genai.upload_file(path=video_path)
            uploaded_files[video_path] = uploaded_obj
            save_uploaded_files(uploaded_files)
            time.sleep(5)
        # else:
            # print(f"Uploaded object for {video_path} is valid.")
    else:
        uploaded_obj = genai.upload_file(path=video_path)
        uploaded_files[video_path] = uploaded_obj
        save_uploaded_files(uploaded_files)
        time.sleep(5)

    return uploaded_obj


def is_uploaded_obj_valid(uploaded_obj, genai):
    """
    Checks the validity of an uploaded object using the genai API.

    Args:
        uploaded_obj: The uploaded file object.
        genai: An object with a `get_file_info` method.

    Returns:
        True if the object is valid, False otherwise.
    """
    try:
        active_list = [file.name for file in genai.list_files() if file.state == 2]
        # Check for successful retrieval and other relevant info.
        if uploaded_obj.name in active_list:
            return True
        else:
            return False

    except Exception as e:
        print(f"Error checking file validity: {e}")
        return False #Return false on error.