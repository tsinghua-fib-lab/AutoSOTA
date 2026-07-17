import json
import pandas as pd
import os
import yaml
import re
import copy

from llm_wrapper import LLMWrapper


def open_raw_free_news_dataset(folder_path: str) -> pd.DataFrame:
    """
    Opens the raw FreeNews dataset from a folder.

    Args:
        folder_path (str): Path to the folder containing the JSON files.

    Returns:
        list of string: List the body of all the gathered texts in the dataset.
    """
    gathered_texts = []
    for i, file_name in enumerate(os.listdir(folder_path)):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                data = json.load(f)

                if data.get("language") == "english":
                    site_name = data.get("thread", {}).get("site_full", "").lower()
                    print(" >> Site name:", site_name)                
                    # language = data.get("language", "").lower()
                    # print(" >> Language:", language)
                    # text_body = data.get("text", "")
                    # print(" >> Text body preview:", text_body[:200])
                    # print()



                    # Extract the text body only
                    text_body = data.get("text", "")
                    gathered_texts.append(text_body)

    return gathered_texts


def open_raw_latest_news_dataset(folder_path: str) -> list[str]:

    with open(folder_path + "Latest_News.json", 'r') as f:
        data = json.load(f)

    df_data = pd.DataFrame(data)
    print(" >> Columns in the dataset:", df_data.columns.tolist())

    # Clean None entries in 'content' and 'link' columns
    df_data = df_data.dropna(subset=["content", "link"])
    df_data = df_data[df_data["content"].notnull() & df_data["link"].notnull()]

    # To filter all the English native domains
    regex_pattern = r"https?://[^/\s]+\.(?:uk|us|au|ca|nz|ie|in|sg|za)(?:/|$)"
    # To filter only metro.co.uk articles
    regex_pattern = r"https?://(?:www\.)?metro\.co\.uk(?:/|$)[^\s]*"
    df_data = df_data[df_data["link"].apply(lambda link: bool(re.search(regex_pattern, link)) if isinstance(link, str) else False)]

    df_data = df_data.drop_duplicates(subset="link", keep="first")

    def remove_picture_annotations(text):
        # Remove patterns like (Picture: ...)
        cleaned_text = re.sub(r" \(Picture:.*?\)", ".", text)
        return cleaned_text

    df_data["content"] = df_data["content"].apply(remove_picture_annotations)

    def drop_first_picture_caption(text):
        # Remove the first line if it starts with "Picture:"
        lines = text.split('. ')
        return ". ".join(lines[1:]).strip()
    
    df_data["content"] = df_data["content"].apply(drop_first_picture_caption)

    def remove_curly(text):
        # Remove text within curly braces {} calls for videos, API etc.
        return re.sub(r" \{.*?\}", "", text)

    df_data["content"] = df_data["content"].apply(remove_curly)

    gathered_texts = df_data["content"].dropna().tolist()

    return gathered_texts
    
    



def filtering_body_of_text(texts):
    """
        Filters out undesired parts of the text body.
    """
    undesired_strings = ["www", "http", "https", "@", "twitter", "facebook", "instagram", "tiktok", "linkedin", "youtube", "reddit", "[", "]", "{", "}", "(", ")", "*", "_", "~", "`", "#", "%", "^", "&", "+", "=", "<", ">", "|", "\\", "/", "$", "\\n"]
    filtered_texts = []
    for i, text_body in enumerate(texts):
        if not any(undesired in text_body.lower() for undesired in undesired_strings):
            filtered_texts.append(text_body)
    return filtered_texts


def compiling_dataframe_infos(gathered_texts, param_file="param.yaml"):
    """
        Compiles the gathered texts into a DataFrame after processing.
    """

    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)

    llm = LLMWrapper(
        hf_token = params["hf_token"],
        **params["model_arguments"]
    )

    text_string_list = []
    completion_text_list = []
    text_string_lengths = []
    text_token_lengths = []
    first_sentence_token_ids = []
    first_sentence_token_lengths = []
    first_sentence_list = []
    for i, text_body in enumerate(gathered_texts):
        if params["data_arguments"].get("truncate_input_words", None) is not None:
            words = text_body.split(' ')
            first_sentence = ' '.join(words[:params["data_arguments"]["truncate_input_words"]])
            rest_of_text = ' '.join(words[params["data_arguments"]["truncate_input_words"]:])
        else:        
            first_sentence = text_body.split('. ')[0] + "."
            rest_of_text = '. '.join(text_body.split('. ')[1:])
        # print(f"Text {i}: {first_sentence}")
        # Check length of the first sentence
        if len(first_sentence) < 200:           # les(first_sentence) > 20:
            # Check total text token length
            encoded_text = llm.tokenizer(text_body, return_tensors="pt", padding=False, truncation=False)
            text_token_length = len(encoded_text["input_ids"][0])

            # Get input text (first sentence) token length
            encoded_first_sentence = llm.tokenizer(first_sentence, return_tensors="pt", padding=False, truncation=False)
            first_sentence_token_length = len(encoded_first_sentence["input_ids"][0])

            if text_token_length - first_sentence_token_length > params["generation_arguments"]["max_new_tokens"]:

                total_allowed_tokens = first_sentence_token_length + params["generation_arguments"]["max_new_tokens"]
                truncated_text_as_token_max = encoded_text["input_ids"][0][:total_allowed_tokens].tolist()
                # print(truncated_text_as_token_max)
                truncated_text_as_token_max = llm.decode(truncated_text_as_token_max, skip_special_tokens=True)
                # print(" >> Truncated text as per max_new_tokens:", truncated_text_as_token_max)

                text_string_list.append(text_body)
                # completion_text_list.append(rest_of_text)
                completion_text_list.append(truncated_text_as_token_max)
                text_string_lengths.append(len(text_body))
                text_token_lengths.append(text_token_length)
                first_sentence_token_ids.append(encoded_first_sentence["input_ids"][0])
                first_sentence_token_lengths.append(first_sentence_token_length)
                first_sentence_list.append(first_sentence)

    # Character Length statistics
    print(" >> Stats about text lengths:")
    print("Total number of files:", len(text_string_list))
    print("Average text length:", sum(text_string_lengths) / len(text_string_lengths) if text_string_lengths else 0)
    print("Max text length:", max(text_string_lengths) if text_string_lengths else 0)
    print("Min text length:", min(text_string_lengths) if text_string_lengths else 0)

    # Token Length statistics
    print(" >> Stats about token lengths:")
    print("Total number of files:", len(text_token_lengths))
    print("Average token length:", sum(text_token_lengths) / len(text_token_lengths) if text_token_lengths else 0)
    print("Max token length:", max(text_token_lengths) if text_token_lengths else 0)
    print("Min token length:", min(text_token_lengths) if text_token_lengths else 0)

    # Now the lists are all the data desired. It will be saved as a dataframe for simulated Vanilla data generation, And as a raw input text file for phrase seeds.
    human_text_dict = []
    for i in range(len(text_string_list)):
        human_text_dict.append({
            "input_text": first_sentence_list[i],
            "input_text_id": i,
            "input_token_length": first_sentence_token_lengths[i],
            "input_token_ids": first_sentence_token_ids[i],
            "output_text": completion_text_list[i],
            "steering_noise": 0.0,
            "steering_type": "human",
            "steering_layers": [],
        })

    df_human = pd.DataFrame(human_text_dict)

    # print(df_human.head())
    return df_human


def get_human_texts(params):
    file_path = 'text_data/newsdataio/'
    gathered_texts = open_raw_latest_news_dataset(file_path)

    print(len(gathered_texts), "texts gathered from the dataset.")

    # filtered_texts = filtering_body_of_text(gathered_texts)

    # print(len(filtered_texts), "texts remaining after filtering.")

    dataframe = compiling_dataframe_infos(gathered_texts)

    human_params = copy.deepcopy(params)
    human_params["steering_arguments"]["noise_max"] = 0.0
    human_params["steering_arguments"]["noise_type"] = "human"
    human_params["steering_arguments"]["steering_layers"] = []


    return dataframe, human_params


def save_input_texts(texts, file_path):
    with open(file_path, 'w') as f:
        for text in texts:
            f.write(text + "\n")
    print(f"Input texts saved to {file_path}")

if __name__ == "__main__":
    ##############
    ## From newsdata.io circa 2021 news
    ##############
    # file_path = 'text_data/newsdataio/'
    # gathered_texts = open_raw_latest_news_dataset(file_path)
    # # filtered_texts = filtering_body_of_text(gathered_texts)
    # dataframe = compiling_dataframe_infos(gathered_texts)
    # # save_input_texts(dataframe["input_text"], "text_data/metro_news_v3.254")

    ##############
    ## Fresher news dataset Webz.io circa jan 2026 news 
    ##############
    file_path = 'text_data/FreeFreshNews/Lifestyle and Leisure_positive_20260104073514'
    gathered_texts = open_raw_free_news_dataset(file_path)



