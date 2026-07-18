"""
Token Counter Utility for SafeFlow

Provides consistent token counting across all components using tiktoken.
Used for both turn-level and session-level token tracking.
"""

import tiktoken
import json
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Universal token counter for SafeFlow.

    Uses tiktoken with a standard encoding to provide consistent
    token counts across all components.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter with specified encoding.

        cl100k_base is used by GPT-4, GPT-3.5-turbo and provides
        a good standard for token counting.
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.encoding_name = encoding_name
        except Exception as e:
            logger.warning(f"Failed to load encoding {encoding_name}: {e}")
            # Fallback to a simple character-based approximation
            self.encoding = None
            self.encoding_name = "character_fallback"

    def count_text_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0

        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Token encoding failed: {e}, using fallback")

        # Fallback: rough approximation (4 chars = 1 token)
        return max(1, len(text) // 4)

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Count tokens in a single message.

        Handles different message types including tool calls.
        """
        tokens = 0

        # Count content
        content = message.get("content", "")
        if content:
            tokens += self.count_text_tokens(str(content))

        # Count role (small overhead)
        role = message.get("role", "")
        tokens += self.count_text_tokens(role)

        # Count tool calls if present
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                # Count function name and arguments
                function_info = tool_call.get("function", {})
                function_name = function_info.get("name", "")
                arguments = function_info.get("arguments", "")

                tokens += self.count_text_tokens(function_name)
                tokens += self.count_text_tokens(str(arguments))

        # Count tool call ID and other metadata
        for key in ["tool_call_id", "name"]:
            if key in message:
                tokens += self.count_text_tokens(str(message[key]))

        # Add small overhead for message structure
        tokens += 4  # Message overhead

        return tokens

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens in a list of messages."""
        return sum(self.count_message_tokens(msg) for msg in messages)

    def count_openai_response_tokens(self, response) -> Dict[str, int]:
        """
        Count tokens in OpenAI response.

        Extracts both prompt tokens and completion tokens if available.
        """
        token_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        try:
            # Try to get usage information from response
            if hasattr(response, 'usage'):
                usage = response.usage
                token_info["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0)
                token_info["completion_tokens"] = getattr(usage, 'completion_tokens', 0)
                token_info["total_tokens"] = getattr(usage, 'total_tokens', 0)
                return token_info
        except Exception as e:
            logger.debug(f"Could not extract usage from response: {e}")

        # Fallback: count manually from response content
        try:
            choice = response.choices[0] if response.choices else None
            if choice and hasattr(choice, 'message'):
                message = choice.message
                content = getattr(message, 'content', '') or ''
                tool_calls = getattr(message, 'tool_calls', []) or []

                completion_tokens = self.count_text_tokens(content)

                # Count tool calls
                for tool_call in tool_calls:
                    if hasattr(tool_call, 'function'):
                        func = tool_call.function
                        completion_tokens += self.count_text_tokens(getattr(func, 'name', ''))
                        completion_tokens += self.count_text_tokens(getattr(func, 'arguments', ''))

                token_info["completion_tokens"] = completion_tokens
                token_info["total_tokens"] = completion_tokens  # We don't have prompt tokens in fallback
        except Exception as e:
            logger.debug(f"Could not count response tokens manually: {e}")

        return token_info


class TokenTracker:
    """
    Session-level token tracking for SafeFlow.

    Tracks both turn-level and cumulative token usage.
    """

    def __init__(self):
        self.counter = TokenCounter()
        self.session_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "turn_count": 0,
            "turns": []  # List of per-turn statistics
        }

    def start_turn(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Start a new turn and calculate prompt tokens.

        Returns turn info with prompt token count.
        """
        prompt_tokens = self.counter.count_messages_tokens(messages)

        turn_info = {
            "turn": len(self.session_stats["turns"]) + 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens
        }

        self.session_stats["turns"].append(turn_info)
        return turn_info

    def end_turn(self, response, response_message: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        End current turn and update token counts.

        Returns complete turn statistics.
        """
        if not self.session_stats["turns"]:
            logger.error("end_turn called without start_turn")
            return {}

        current_turn = self.session_stats["turns"][-1]

        # Count completion tokens
        response_tokens = self.counter.count_openai_response_tokens(response)

        # If we have the response message, count it manually as backup
        if response_message and response_tokens["completion_tokens"] == 0:
            response_tokens["completion_tokens"] = self.counter.count_message_tokens(response_message)

        # Update current turn
        current_turn["completion_tokens"] = response_tokens["completion_tokens"]
        current_turn["total_tokens"] = current_turn["prompt_tokens"] + current_turn["completion_tokens"]

        # Update session totals
        self.session_stats["total_prompt_tokens"] += current_turn["prompt_tokens"]
        self.session_stats["total_completion_tokens"] += current_turn["completion_tokens"]
        self.session_stats["total_tokens"] += current_turn["total_tokens"]
        self.session_stats["turn_count"] = len(self.session_stats["turns"])

        return current_turn

    def get_session_stats(self) -> Dict[str, Any]:
        """Get complete session statistics."""
        return self.session_stats.copy()

    def get_current_turn_prompt_tokens(self) -> int:
        """Get prompt tokens for current turn (if in progress)."""
        if self.session_stats["turns"]:
            return self.session_stats["turns"][-1]["prompt_tokens"]
        return 0

    def should_summarize_context(self, threshold: int = 8000) -> bool:
        """
        Check if context should be summarized based on token count.

        Returns True if current turn prompt tokens exceed threshold.
        """
        return self.get_current_turn_prompt_tokens() > threshold


# Global instance for easy access
default_token_tracker = TokenTracker()