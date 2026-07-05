import os

from model_api import query_llm_single

import dotenv
dotenv.load_dotenv()


def test_gemini_single():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set; skipping Gemini single test.")
        return
    response = query_llm_single(
        model="google/gemini-3-pro-preview",
        message="Hello from Gemini single-call test!",
        prompt="You are a helpful assistant.",
        temperature=0.2,
        reasoning="high",
    )
    print(response)
    assert isinstance(response, str)
    assert response
    print("Gemini single API test passed.")


if __name__ == "__main__":
    test_gemini_single()
