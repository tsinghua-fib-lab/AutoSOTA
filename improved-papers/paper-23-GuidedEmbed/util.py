import openai
from openai import OpenAI
import time
import threading
from typing import List

# API configuration
api_key = "api_key"
openai.api_key = api_key
client = OpenAI(api_key=api_key)

def get_completion(prompt, model="gpt-4o-mini", temperature=0.7):
    """
    Call OpenAI API to get response
    
    Args:
        prompt (str): Input prompt
        model (str): Model name to use
        temperature (float): Temperature parameter to control randomness of output
        
    Returns:
        str: Model's response content
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

def monitor_thread(threads):
    while any(t.is_alive() for t in threads):
        # Subtracting 1 is to exclude the monitor thread itself
        # As the main thread calculated, it will have one more thread than the actual child thread
        print(f"Number of active threads: {threading.active_count() - 1}")
        time.sleep(1)


def thread_response(prompt_list: List[str],
                   model: str = "gpt-4o-mini", 
                   temperature: float = 0.7,
                   max_threads: int = 80, 
                   max_retries: int = 3):
    """
    Input a prompt_list and get response list from openAI api
    """
    result_list = []
    thread_list = []

    semaphore = threading.Semaphore(max_threads)

    # wrap thread with max thread limit
    def thread_wrapper(id, message: str = "", max_retries: int = 3):
        with semaphore:
            get_thread_response(id, message, max_retries)

    # wrap thread with max retry times
    def get_thread_response(id, message: str = "", max_retries: int = 3):
        retries = 0
        while retries < max_retries:
            try:
                print(f"Thread {id} starting.")
                response = get_completion(message, model=model, temperature=temperature)
                result_list.append({
                    "index": id,
                    "response": response
                })
                print(f"Thread {id} finishing.")
                break
            except Exception as e:
                retries += 1
                print(f"Thread {id} failed: {e} (Attempt {retries}/{max_retries})")
                if retries < max_retries:
                    print(f"The thread {id} will restart in 60 seconds ")
                    time.sleep(60)  # wait 60 seconds
                    print(f"Restarting thread {id}")
                else:
                    result_list.append({
                        "index": id,
                        "response": 'none'
                    })
                    print(f"Thread {id} reached maximum retry attempts and will not be restarted")

    for i in range(len(prompt_list)):
        t = threading.Thread(target=thread_wrapper, args=(i, prompt_list[i], max_retries))
        thread_list.append(t)

    for t in thread_list:
        t.start()

    monitor = threading.Thread(target=monitor_thread, args=(thread_list,))
    monitor.start()

    for t in thread_list:
        t.join()

    monitor.join()
    
    # Sort by index and return only response content
    result_list = sorted(result_list, key=lambda item: item["index"])
    assert len(result_list) == len(prompt_list)
    return [item["response"] for item in result_list]