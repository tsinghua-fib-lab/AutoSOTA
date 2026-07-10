from typing import Dict, Any, Optional, Union
from .generator import DirectGenerator, MultimodalGenerator, EmbeddingGenerator


def create_generator(model_config: Dict[str, Any],
                    cache_config: Optional[Dict[str, Any]] = None) -> Union[DirectGenerator, MultimodalGenerator, EmbeddingGenerator]:
    """Factory function to create generator instances from configuration

    Args:
        model_config: Model configuration dictionary containing:
            - generator_type: Optional["direct", "multimodal", "embedding"] - defaults to "direct"
            - api_model_name: API model name/ID for provider calls
            - name: User-defined name for caching and reference
            - base_url: API base URL
            - api_key: API key
            - temperature: Optional temperature setting
            - top_p: Optional top_p setting
            - timeout: Optional timeout setting
            - reasoning_effort: Optional reasoning effort level
            - extra_body: Optional extra body parameters
            - pricing: Optional pricing configuration
        cache_config: Optional cache configuration

    Returns:
        DirectGenerator, MultimodalGenerator, or EmbeddingGenerator instance based on generator_type
    """
    generator_type = model_config.get("generator_type", "direct").lower()
    
    # Common parameters for both generator types
    common_params = {
        "model_name": model_config["api_model_name"],
        "config_name": model_config.get("name", model_config["api_model_name"]),  # Use name for cache, fallback to api_model_name
        "base_url": model_config["base_url"],
        "api_key": model_config["api_key"],
        "temperature": model_config.get("temperature", 0.0),
        "top_p": model_config.get("top_p", 1.0),
        "timeout": model_config.get("timeout", 500),
        "reasoning_effort": model_config.get("reasoning_effort"),
        "extra_body": model_config.get("extra_body", {}),
        "cache_config": cache_config,
        "pricing_config": model_config.get("pricing", {}),
    }
    # Some parameters without default values
    if "max_tokens" in model_config:
        common_params.update({"max_tokens": model_config["max_tokens"]})
    if "stream" in model_config:
        common_params.update({"stream": model_config["stream"]})
    if generator_type == "multimodal":
        return MultimodalGenerator(**common_params)
    elif generator_type == "embedding":
        # EmbeddingGenerator uses fewer parameters
        embedding_params = {
            "model_name": model_config["api_model_name"],
            "config_name": model_config.get("name", model_config["api_model_name"]),
            "base_url": model_config["base_url"],
            "api_key": model_config["api_key"],
            "timeout": model_config.get("timeout", 500),
            "cache_config": cache_config,
            "max_context_length": model_config.get("max_context_length"),
            "model_path": model_config.get("model_path"),
        }
        return EmbeddingGenerator(**embedding_params)
    elif generator_type == "direct":
        return DirectGenerator(**common_params)
    else:
        raise ValueError(
            f"Unknown generator_type: {generator_type}. "
            f"Supported types: 'direct', 'multimodal', 'embedding'"
        )
