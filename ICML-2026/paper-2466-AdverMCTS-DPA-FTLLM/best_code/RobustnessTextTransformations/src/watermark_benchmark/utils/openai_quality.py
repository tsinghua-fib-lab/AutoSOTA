""" Query OpenAI APIs to use GPT as a LLM evaluator """

import math
import multiprocessing
import os
import re
import signal
import sys
import time
from dataclasses import replace

from openai import OpenAI
from tqdm import tqdm

from ..metrics.llm_rating import prompt_no_cot as rating_prompt
from . import (
    get_input_file,
    get_output_file,
    load_config,
    setup_randomness,
)
from .classes import Generation

rating_prompt = (
    rating_prompt.replace(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", ""
    )
    .replace("<|start_header_id|>user<|end_header_id|> ", "\n")
    .replace(
        "<|eot_id|> <|start_header_id|>assistant<|end_header_id|> ", "\n\n"
    )
)

rating_prompt = (
    rating_prompt.replace("[INST]", "")
    .replace("[/INST]", "\n\n")
    .replace("<<SYS>>", "")
    .replace("<</SYS>>", "\n\n")
)

global_process_list = []


def rate_completions(
    prompts,
    responses,
    task_queue=None,
    response_queue=None,
    number_of_processes=4,
    display_progress=True,
    model="gpt-3.5-turbo",
):
    """
    Rates the quality of responses to a given set of prompts.

    Args:
        prompts (list or str): A list of prompts or a single prompt string.
        responses (list or str): A list of responses or a single response string.
        task_queue (multiprocessing.Queue): A queue for tasks to be processed.
        response_queue (multiprocessing.Queue): A queue for responses to be collected.
        number_of_processes (int): The number of processes to use for rating.

    Returns:
        list or float: A list of ratings or a single rating if a single prompt was provided.
    """
    if len(prompts) != len(responses):
        raise ValueError("The number of prompts and responses must be equal.")

    if not isinstance(prompts, list):
        prompts = [prompts]
        responses = [responses]
        return_single = True
    else:
        return_single = False

    if task_queue is None or response_queue is None:
        task_queue, manager = init_servers(
            number_of_processes=number_of_processes
        )
        response_queue = manager.Queue()
        self_hosted = True
    else:
        self_hosted = False

    tasks = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        tasks.append(
            (
                i,
                rating_prompt.format(prompt, response),
                2048,
                {"temperature": 0, "model": model, "timeout": 30, "stop": "]]"},
                response_queue,
            )
        )

    for task in tasks:
        task_queue.put(task)

    ratings = [0 for _ in prompts]
    for _ in tqdm(
        prompts,
        total=len(prompts),
        desc="Rating responses",
        disable=not display_progress,
    ):
        i, resp = response_queue.get(block=True)
        try:
            ratings[i] = (
                float(
                    re.search(
                        r"\[\[[0-9][0-9.]?(/10)?\]*",
                        resp.choices[0].message.content.strip(),
                    )
                    .group(0)
                    .split("/")[0]
                    .replace("[", "")
                    .replace("]", "")
                )
                / 10
            )
            print(
                f"{i}:\n\n{tasks[i][1]}\n\n###\n\n{resp.choices[0].message.content}\n\n###\n\n{ratings[i]}"
            )
            print("\n\n\n#############\n\n\n")
            if ratings[i] > 1:
                print(
                    "Warning: Rating > 1: {} (input {})".format(
                        ratings[i], resp.choices[0].message.content.strip()
                    )
                )
                ratings[i] = 1
        except AttributeError:
            ratings[i] = 0
            # kill_servers()
            # raise ValueError(
            #     "Error with rating: response is {}: ".format(
            #         resp.choices[0].message.content
            #     )
            # ) from e

    if self_hosted:
        kill_servers()

    return ratings[0] if return_single else ratings


def openai_chat_server(call_queue, leader=False):
    """
    A function that listens to a call queue for incoming tasks, and processes them using OpenAI's API.

    Args:
    call_queue (Queue): A queue object that contains incoming tasks. These are made of the following elements:
        id: id for this task.
        message: a string representing the user's message prompt.
        max_tokens: an integer representing the maximum number of tokens to generate.
        kwargs: a dictionary containing optional keyword arguments to be passed to the call_openai function.
        dest_queue: a queue object where the result of the task will be put.

    Returns:
        None
    """
    client = OpenAI()

    while True:
        task = call_queue.get(block=True)
        if task is None:
            return

        compl_id, message, max_tokens, kwargs, dest_queue = task
        rslt = call_openai(client, message, max_tokens, **kwargs)
        if rslt == 0 and not leader:
            call_queue.put(task)
            print("Reducing the number of OpenAI threads due to Rate Limit")
            return
        elif rslt == 0 and leader:
            call_queue.put(task)
        else:
            dest_queue.put((compl_id, rslt))


def call_openai(
    client,
    message,
    max_tokens,
    query_type="chat",
    model="gpt-3.5-turbo",
    temperature=1.0,
    top_p=1,
    presence_penalty=0,
    frequency_penalty=0,
    system_prompt=None,
    stop=None,
    timeout=None,
    n=1,
):
    """
    Calls the OpenAI API to generate text based on the given parameters.

    Args:
        client (openai.api_client.Client): The OpenAI API client.
        message (str): The user's message prompt.
        max_tokens (int): The maximum number of tokens to generate.
        query_type (str): The type of completion to use. Defaults to "chat".
        model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Controls the "creativity" of the generated text. Higher values result in more diverse text. Defaults to 1.0.
        top_p (float, optional): Controls the "quality" of the generated text. Higher values result in higher quality text. Defaults to 1.
        presence_penalty (float, optional): Controls how much the model avoids repeating words or phrases from the prompt. Defaults to 0.
        frequency_penalty (float, optional): Controls how much the model avoids generating words or phrases that were already generated in previous responses. Defaults to 0.
        system_prompt (str, optional): A prompt to be included before the user's message prompt. Defaults to None.
        stop (str, optional): A stop sequence
        timeout (int, optional): The maximum time to wait for a response from the API, in seconds. Defaults to 10.
        n (int, optional): The number of responses to generate. Defaults to 1.

    Returns:
        The generated responses from the OpenAI API.
    """

    def loop(f, params):
        retry = 0
        while retry < 7:
            try:
                return f(params)
            except Exception as e:
                if retry > 5:
                    print(f"Error {retry}: {e}\n{params}")
                if "maximum context length" in str(e):
                    print("Context length exceeded")
                    return None
                if (
                    "Rate limit" in str(e)
                    or "overloaded" in str(e)
                    or "timed out" in str(e)
                ):
                    if "timed out" in str(e) and retry < 2:
                        params["timeout"] += 30 * retry
                    elif retry < 1:
                        time.sleep(30 * (1 + retry))
                    else:
                        print(e)
                        return 0

                else:
                    time.sleep(3 * retry)
                retry += 1
                continue
        return None

    request_params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop,
    }

    if max_tokens != math.inf:
        request_params["max_tokens"] = max_tokens

    if timeout is not None:
        request_params["timeout"] = timeout

    if query_type == "chat":
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(message)},
            ]
        else:
            messages = [{"role": "user", "content": message}]
        request_params["messages"] = messages
        return loop(
            lambda x: client.chat.completions.create(**x), request_params
        )

    request_params["prompt"] = message
    return loop(lambda x: client.completions.create(**x), request_params)


def init_servers(number_of_processes=4):
    """
    Initializes multiple chat servers using multiprocessing.

    Args:
        number_of_processes (int): The number of server processes to start. Default is 4.

    Returns:
        tuple: A tuple containing a call queue and a global manager object.
    """
    global_manager = multiprocessing.Manager()
    call_queue = global_manager.Queue()

    for i in range(number_of_processes):
        p = multiprocessing.Process(
            target=openai_chat_server, args=(call_queue, i == 0)
        )
        p.start()
        global_process_list.append(p)

    return call_queue, global_manager


def kill_servers():
    """
    Kill all processes
    """
    for p in global_process_list:
        p.terminate()
        p.join()


def standalone_server(inputs, **kwargs):
    """
    Run a standalone server to process inputs and return responses.

    Args:
        inputs: A string or a list of strings representing the inputs to be processed.
        **kwargs: Additional keyword arguments for server configuration.

    Returns:
        If `inputs` is a string, returns a single response string.
        If `inputs` is a list of strings, returns a list of response strings.
    """
    queue, mgr = init_servers()
    resp_queue = mgr.Queue()
    kwargs["timeout"] = 60
    if isinstance(inputs, str):
        inputs_mod = [inputs]
    else:
        inputs_mod = inputs
    for idx, input in enumerate(inputs_mod):
        queue.put((idx, input, math.inf, kwargs, resp_queue))
    responses = ["" for _ in inputs_mod]
    for _ in inputs_mod:
        idx, resp = resp_queue.get(block=True)
        responses[idx] = (
            resp.choices[0].message.content
            if "chat" not in kwargs or kwargs["query_type"] == "chat"
            else resp.choices[0].text
        )
    kill_servers()
    if isinstance(inputs, str):
        return responses[0]
    return responses


def graceful_exit(sig, frame):
    """
    Kill all processes on SIGINT
    """
    kill_servers()
    exit()


signal.signal(signal.SIGINT, graceful_exit)


def run(config_file, model="gpt-3.5-turbo"):
    # load config
    config = (
        load_config(config_file)
        if isinstance(config_file, str)
        else config_file
    )
    setup_randomness(config)

    # load generations
    generations = Generation.from_file(get_input_file(config))
    outfilepath = get_output_file(config)
    if not os.path.exists(outfilepath):
        Generation.to_file(outfilepath)
    existing = {
        str(
            (
                (
                    g.watermark.to_dict(True, True)
                    if g.watermark is not None
                    else g.temp
                ),
                g.id,
                g.attack,
            )
        )
        for g in Generation.from_file(outfilepath)
    }
    tasks = [
        g
        for g in generations
        if str(
            (
                (
                    g.watermark.to_dict(True, True)
                    if g.watermark is not None
                    else g.temp
                ),
                g.id,
                g.attack,
            )
        )
        not in existing
    ]

    if not len(tasks):
        return

    output_file = get_output_file(config)

    if not os.path.exists(output_file):
        with open(output_file, "w") as outfile:
            pass

    prompts, responses, filt = [], [], []
    for t in tasks:
        if t.id % 100 > 33:
            continue
        filt.append(t)
        cleaned_prompt = re.sub(
            r"<<SYS>>.*<<\/SYS>>",
            "",
            t.prompt.replace("[/INST]", "").replace("[INST]", ""),
        ).strip()
        prompts.append(cleaned_prompt)
        responses.append(t.response)

    ratings = rate_completions(
        prompts,
        responses,
        model=model,
        number_of_processes=config.openai_parallelism,
    )

    with open(outfilepath, "a") as outfile:
        for i, t in enumerate(filt):
            r = ratings[i]
            t = replace(t, rating=r)
            tasks[i] = t
        outfile.write("\n".join(str(gen) for gen in tasks) + "\n")


def main():
    config = load_config(sys.argv[1])
    model = sys.argv[2]
    config.input_file = sys.argv[3]
    config.output_file = sys.argv[4]
    run(config, model=model)
