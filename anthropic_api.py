#!/usr/bin/env python3
"""
Anthropic API implementation for LLM queries.
Provides LLM query functions using Anthropic's public API.
"""

import os
import time
from typing import Any

try:
    import anthropic
    from anthropic import RateLimitError, APIError
except ImportError:
    anthropic = None
    RateLimitError = None
    APIError = None
    print("⚠️  Warning: anthropic package not installed. Install with: pip install anthropic")


def query_llm_with_usage(
    prompt: str,
    system_prompt: str = "You're a helpful assistant, an experienced engineer, and you follow instructions if given.",
    model: str = "claude-sonnet-4-20250514",  # Anthropic's latest model
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Query the Anthropic API and return response with token usage information.

    Args:
        prompt: The user's question or instruction
        system_prompt: Instructions for how the AI should behave
        model: Which AI model to use (Anthropic model names)
        max_tokens: Maximum length of response
        max_retries: Number of retry attempts on failure

    Returns:
        Dictionary containing:
            - response: The AI's response text
            - prompt_tokens: Number of tokens in the input
            - completion_tokens: Number of tokens in the output
            - total_tokens: Total tokens used
            - failed: bool indicating if API call failed
            - estimated: Optional bool if tokens are estimated
            - error: Optional error message if failed

    Example:
        >>> result = query_llm_with_usage("What is 2+2?")
        >>> print(f"Answer: {result['response']}")
        >>> print(f"Tokens used: {result['total_tokens']}")
    """
    # Check for Anthropic package
    if anthropic is None:
        raise ImportError(
            "anthropic package is required but not installed. "
            "Install with: pip install anthropic"
        )

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key from https://console.anthropic.com/"
        )

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    last_error = None

    for attempt in range(max_retries):
        try:
            # Make API call
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=60.0,  # 60 second timeout
            )

            # Validate response structure before accessing
            if not message.content or len(message.content) == 0:
                raise RuntimeError("API returned empty content array")

            # Extract response text safely
            response_text = message.content[0].text
            if response_text is None:
                raise RuntimeError("API returned None for response text")

            # Build usage info
            usage_info: dict[str, Any] = {
                "response": response_text,
                "failed": False  # Track successful API call
            }

            # Anthropic API provides usage information
            if hasattr(message, 'usage'):
                usage_info["prompt_tokens"] = message.usage.input_tokens
                usage_info["completion_tokens"] = message.usage.output_tokens
                usage_info["total_tokens"] = message.usage.input_tokens + message.usage.output_tokens
            else:
                # Fallback: estimate tokens (rough approximation: ~4 chars per token)
                usage_info["prompt_tokens"] = (len(system_prompt) + len(prompt)) // 4
                usage_info["completion_tokens"] = len(response_text) // 4
                usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
                usage_info["estimated"] = True

            return usage_info

        except RateLimitError as e:
            # Handle rate limit errors with longer backoff
            last_error = e
            error_str = str(e)[:100]
            print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries}): {error_str}")

            if attempt < max_retries - 1:
                # Longer backoff for rate limits: 5s, 10s, 20s
                wait_time = (2 ** attempt) * 5
                print(f"    Waiting {wait_time}s before retry (rate limit backoff)...")
                time.sleep(wait_time)

        except Exception as e:
            # Handle other errors with standard backoff
            last_error = e
            error_str = str(e)[:100]
            print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {error_str}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Standard exponential backoff: 1s, 2s, 4s
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    # All retries failed - return a default fallback response
    print(f"⚠️  All {max_retries} attempts failed. Using default response. Last error: {str(last_error)[:200]}")
    usage_info = {
        "response": '{"reasoning": "API error after all retries - using default", "decision": "accept", "offer": 50}',
        "prompt_tokens": (len(system_prompt) + len(prompt)) // 4,
        "completion_tokens": 50,
        "total_tokens": (len(system_prompt) + len(prompt)) // 4 + 50,
        "estimated": True,
        "failed": True,  # Mark this API call as failed
        "error": str(last_error)[:200]
    }
    return usage_info


def query_with_retry(
    prompt: str,
    system_prompt: str = "You're a helpful assistant.",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """
    Query Anthropic API with automatic retry on failure.

    Args:
        prompt: The user's question or instruction
        system_prompt: Instructions for how the AI should behave
        model: Which AI model to use
        max_tokens: Maximum length of response
        max_retries: Number of retry attempts on failure

    Returns:
        The AI's response as a string

    Raises:
        RuntimeError: If all retry attempts fail

    Example:
        >>> answer = query_with_retry("Explain quantum computing", max_retries=3)
    """
    # Check for Anthropic package
    if anthropic is None:
        raise ImportError(
            "anthropic package is required but not installed. "
            "Install with: pip install anthropic"
        )

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key from https://console.anthropic.com/"
        )

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    last_error = None

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=60.0,
            )

            # Validate response structure before accessing
            if not message.content or len(message.content) == 0:
                raise RuntimeError("API returned empty content array")

            result = message.content[0].text
            if result is None:
                raise RuntimeError("API returned None")

            return result

        except RateLimitError as e:
            # Handle rate limit errors with longer backoff
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s for rate limits
                print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Standard exponential backoff

    raise RuntimeError(f"All {max_retries} attempts failed. Last error: {last_error}")


def query_llm(
    prompt: str,
    system_prompt: str = "You're a helpful assistant, an experienced engineer, and you follow instructions if given.",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> str:
    """
    Simple query to Anthropic API (single attempt).

    Args:
        prompt: The user's question or instruction
        system_prompt: Instructions for how the AI should behave
        model: Which AI model to use
        max_tokens: Maximum length of response

    Returns:
        The AI's response as a string

    Example:
        >>> answer = query_llm("What is the capital of France?")
        >>> print(answer)
    """
    # Check for Anthropic package
    if anthropic is None:
        raise ImportError(
            "anthropic package is required but not installed. "
            "Install with: pip install anthropic"
        )

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get your API key from https://console.anthropic.com/"
        )

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Make API call
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ],
        timeout=60.0,
    )

    # Validate response structure before accessing
    if not message.content or len(message.content) == 0:
        raise RuntimeError("API returned empty content array")

    result = message.content[0].text
    if result is None:
        raise RuntimeError("Anthropic API returned None content")

    return result


# Available Anthropic models (as of Nov 2024)
AVAILABLE_MODELS = [
    "claude-sonnet-4-20250514",     # Latest Sonnet 4
    "claude-3-5-sonnet-20241022",   # Claude 3.5 Sonnet
    "claude-3-opus-20240229",       # Most powerful (but more expensive)
    "claude-3-sonnet-20240229",     # Balanced
    "claude-3-haiku-20240307",      # Fast and cheap
]

DEFAULT_MODEL = "claude-sonnet-4-20250514"


if __name__ == "__main__":
    # Test the API if run directly
    print("Testing Anthropic API...")
    print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    print(f"Default model: {DEFAULT_MODEL}")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ ANTHROPIC_API_KEY not set!")
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
    else:
        print("\n✅ ANTHROPIC_API_KEY is set")

        try:
            # Test a simple query
            result = query_llm_with_usage(
                "What is 2+2? Answer in one word.",
                max_tokens=10
            )
            print(f"\nTest query result:")
            print(f"  Response: {result['response']}")
            print(f"  Tokens used: {result.get('total_tokens', 'N/A')}")
            print(f"  Failed: {result.get('failed', False)}")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")