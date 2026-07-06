import os

from model_api import query_llm_batch

import dotenv
dotenv.load_dotenv()


def test_gemini_batch():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set; skipping Gemini batch test.")
        return
    responses = query_llm_batch(
        model="google/gemini-3-pro-preview",
        messages_list=["Hello world!", "How are you?"],
        prompt="You are a helpful assistant.",
        temperature=0.2,
        reasoning="high",
    )
    print(responses)
    assert isinstance(responses, list)
    assert len(responses) == 2
    for r in responses:
        assert isinstance(r, str)
    print("Gemini batch API test passed.")


test_gemini_batch()
