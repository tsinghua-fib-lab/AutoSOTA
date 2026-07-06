import json
import random
import openai
import time
from timeout_decorator import timeout
from content.utils import *
from database_utils import round_two_decimal, trans_url
import os
from pathlib import Path
from openai import OpenAI
import re

openai.api_key = openai_api_key


def temp_sleep(seconds=1):
    time.sleep(seconds)


def ChatGPT_single_request(prompt):
    temp_sleep()
    client = OpenAI(
        api_key=openai_api_key
        )
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",messages=[{"role": "user", "content": prompt}]
)
    return completion.choices[0].message.content

# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================


def GPT4_request(prompt):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    temp_sleep()
    try:
       
        client = OpenAI(
            api_key=openai_api_key
            )
        completion = client.chat.completions.create(
            model="gpt-4",messages=[{"role": "user", "content": prompt}]
)
       # print(completion)
        return completion.choices[0].message.content
    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"


#@timeout(150)
def ChatGPT_request(prompt):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    try:
       # completion = openai.ChatCompletion.create(
        #    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        #)
        client = OpenAI(
            api_key=openai_api_key
            )
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",messages=[{"role": "user", "content": prompt}]
)
       # print(completion)
        return completion.choices[0].message.content

    except:
        print("ChatGPT ERROR!")
        return "ChatGPT ERROR"

def GPT4o_images_request(prompt, image_url1, image_url2, image_url3 ):#
    try:
        chat_completion = openai.chat.completions.create(
              model="gpt-4o",
              messages=[
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url","image_url": {"url": image_url1,},},
                    {"type": "image_url","image_url": {"url": image_url2,},},
                    {"type": "image_url","image_url": {"url": image_url3,},},
                  ],
                }
              ],
              #max_tokens=300,
            )
        return chat_completion.choices[0].message.content

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"


#@timeout(50)
def send_request(prompt):
    curr_gpt_response = ChatGPT_request(prompt).strip()
    return curr_gpt_response


def GPT4_safe_generate_response(
    prompt,
    example_output,
    special_instruction,
    repeat=3,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += (
        f"Output the response to the prompt above in json. {special_instruction}\n"
    )
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            with eventlet.Timeout(1):
                curr_gpt_response = GPT4_request(prompt).strip()

            end_index = curr_gpt_response.rfind("}") + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except Exception:
            print("GPT connection error, {}".format(Exception))
            pass

    return False


def ChatGPT_safe_generate_response(
    prompt,
    example_output,
    special_instruction,
    repeat=3,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):
    # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += (
        f"Output the response to the prompt above in json. {special_instruction}\n"
    )
    prompt += "Please provide the response in the following format:\n"
    prompt += '{"output": "' + str(example_output) + '"}'


    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)
    # eventlet.monkey_patch()

    for i in range(repeat):
        try:
            #url1=trans_url("plot_stock.jpg")
            #url1=trans_url("plot_stock2.jpg")
            #curr_gpt_response =  GPT4o_3images_request(prompt, url1, url2, url3 ).strip()
            curr_gpt_response = ChatGPT_request(prompt).strip()#.replace("\n","")
            curr_gpt_response = re.sub(r'\s{3,}', '\n', curr_gpt_response).replace("\n","\\n")
            # curr_gpt_response = send_request(prompt)
            #print("curr_gpt_response",curr_gpt_response)
            end_index = curr_gpt_response.rfind("}") + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]


            if verbose:
                print("---GPT Response---")
                print(curr_gpt_response)
                print("---end of GPT Response---")

            print(func_validate(curr_gpt_response, prompt=prompt))
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                # temp_sleep(5)
        # except eventlet.timeout.Timeout:
        except:
            print("GPT connection error, {}".format(Exception))
            pass

    #return False



def ChatGPT_safe_generate_response_OLD(
    prompt,
    repeat=3,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================


def GPT_request(prompt, gpt_parameter):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
    temp_sleep()
    try:
        response = openai.Completion.create(
            model=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"],
        )
        return response.choices[0].text
    except:
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
    """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(round_two_decimal(i)) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(
    prompt,
    gpt_parameter,
    repeat=5,
    fail_safe_response="error",
    func_validate=None,
    func_clean_up=None,
    verbose=False,
):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


if __name__ == "__main__":
    gpt_parameter = {
        "engine": "text-davinci-003",
        "max_tokens": 50,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": ['"'],
    }
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gpt_response):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True

    def __func_clean_up(gpt_response):
        cleaned_response = gpt_response.strip()
        return cleaned_response

    output = safe_generate_response(
        prompt, gpt_parameter, 5, "rest", __func_validate, __func_clean_up, True
    )

    print(output)
