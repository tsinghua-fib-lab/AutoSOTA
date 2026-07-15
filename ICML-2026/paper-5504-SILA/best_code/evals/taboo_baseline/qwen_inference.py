"""
Async Qwen inference using vLLM's OpenAI-compatible server.

This uses async methods so you can fire off multiple requests concurrently,
and vLLM will automatically batch them for efficient processing.

USAGE:
------
1. Start the vLLM server (one time, keep it running):
   $ python start_qwen_server.py --model-size 7b
   
   This will take a few minutes to load the model, but then it stays running.

2. Use the async API in your code:
   
   import asyncio
   from qwen_inference import AsyncQwenInference
   
   async def main():
       async with AsyncQwenInference(model_name="Qwen/Qwen2.5-7B-Instruct") as client:
           # Single request
           answer = await client.generate_single("What is 2+2?")
           
           # Many concurrent requests (vLLM batches them automatically)
           questions = ["Question 1?", "Question 2?", ...]
           answers = await client.generate_batch(questions)
   
   asyncio.run(main())
"""
import asyncio
from typing import Optional
import httpx


class AsyncQwenInference:
    """Async wrapper for Qwen inference via vLLM server."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 300.0,
        system_message: str | None = None,
    ):
        """
        Initialize the async inference client.
        
        Args:
            model_name: Model name (must match what's loaded in vLLM server)
            base_url: Base URL of the vLLM server (OpenAI-compatible API)
            timeout: Request timeout in seconds (default 5 minutes for long generations)
            system_message: Optional system message for all requests
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.system_message = system_message
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Default generation parameters
        self.default_params = {
            "temperature": 0.0,  # Greedy decoding for consistency
            "max_tokens": 256,
            "top_p": 1.0,
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def format_messages(
        self, 
        prompt: str, 
        system_message: str | None = None
    ) -> list[dict[str, str]]:
        """
        Format a prompt using Qwen chat template.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message (overrides default)
            
        Returns:
            List of message dicts for the chat API
        """
        messages = []
        
        sys_msg = system_message or self.system_message
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def generate_single(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response for a single prompt asynchronously.
        
        Args:
            prompt: User prompt
            system_message: Optional system message override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated response (string)
        """
        params = self.default_params.copy()
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        
        messages = self.format_messages(prompt, system_message)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **params
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        if response.status_code != 200:
            # Include response body in error for debugging
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        
        return answer
    
    async def generate_batch(
        self,
        prompts: list[str],
        system_message: str | None = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_concurrent: int = 100,
        **kwargs
    ) -> list[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        This fires off all requests asynchronously, and vLLM will automatically
        batch them for efficient processing.
        
        Args:
            prompts: List of user prompts
            system_message: Optional system message override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            max_concurrent: Maximum concurrent requests (semaphore limit)
            **kwargs: Additional parameters for the API
            
        Returns:
            List of generated responses (strings)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate_single(
                    prompt, system_message, temperature, max_tokens, **kwargs
                )
        
        # Fire off all requests concurrently (with semaphore limiting)
        tasks = [generate_with_semaphore(p) for p in prompts]
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        results = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"Warning: Request {i} failed with error: {resp}")
                results.append("")  # Empty string for failed requests
            else:
                results.append(resp)
        
        return results


async def check_server_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if the vLLM server is running and healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code == 200
    except Exception:
        return False


async def test_inference(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Test the async inference setup with a few examples."""
    print("\n" + "=" * 80)
    print("Testing Async Qwen Inference")
    print("=" * 80 + "\n")
    
    # Check server health first
    if not await check_server_health():
        print("ERROR: vLLM server is not running or not healthy!")
        print("Start it with: python start_qwen_server.py --model-size 7b")
        return
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
    ]
    
    async with AsyncQwenInference(model_name=model_name) as client:
        print("Testing single inference...")
        answer = await client.generate_single(test_prompts[0])
        print(f"Q: {test_prompts[0]}")
        print(f"A: {answer}\n")
        
        print("\n" + "-" * 80)
        print("Running batch inference (all requests fired concurrently)...")
        print("-" * 80 + "\n")
        
        answers = await client.generate_batch(test_prompts)
        
        for prompt, answer in zip(test_prompts, answers):
            print(f"Q: {prompt}")
            print(f"A: {answer}")
            print()
    
    print("=" * 80)
    print("Async inference setup complete and working!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qwen inference via vLLM")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (must match server)"
    )
    args = parser.parse_args()
    
    print("\n" + "!" * 80)
    print("IMPORTANT: Make sure the vLLM server is running first!")
    print("Start it with: python start_qwen_server.py")
    print("!" * 80 + "\n")
    
    asyncio.run(test_inference(args.model))
