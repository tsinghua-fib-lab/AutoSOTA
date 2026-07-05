import os
from model_api import query_llm_batch

import dotenv
dotenv.load_dotenv()

def test_anthropic_batch():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    responses = query_llm_batch(
        model="anthropic/claude-opus-4.1",
        messages_list=["Hello world!", "How are you?"],
        prompt="You are a helpful assistant.",
        temperature=1,
        reasoning="high"
    )
    print(responses)
    assert isinstance(responses, list)
    assert len(responses) == 2
    for r in responses:
        assert isinstance(r, str)
    print("Anthropic batch API test passed.")

test_anthropic_batch()
