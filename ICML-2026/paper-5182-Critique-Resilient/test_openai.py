import os
from model_api import query_llm_batch

import dotenv
dotenv.load_dotenv()

def test_openai_batch():
    api_key = os.getenv("OPENAI_API_KEY")
    responses = query_llm_batch(
        model="openai/gpt-5.1",
        messages_list=["Hello world!", "How are you?"],
        prompt="You are a helpful assistant.",
        temperature=1, # gpt-5 doesn't support temperature 0
        reasoning="high"
    )
    print(responses)
    assert isinstance(responses, list)
    assert len(responses) == 2
    for r in responses:
        assert isinstance(r, str)
    print("OpenAI batch API test passed.")

test_openai_batch()