#!/usr/bin/env python3
"""
LLM API Router

Simple router that selects between different LLM API providers based on
the LLM_PROVIDER environment variable.

Usage:
    # Use Apple's internal API (default)
    python ultimatum_game.py

    # Use Anthropic's public API
    export LLM_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python ultimatum_game.py

This module provides a drop-in replacement for token_counter imports:
    OLD: from token_counter import query_llm_with_usage
    NEW: from llm_api import query_llm_with_usage
"""

import os

# Determine which provider to use
provider = os.getenv("LLM_PROVIDER", "apple").lower()

print(f"🤖 Using LLM provider: {provider}")

if provider == "anthropic":
    # Use Anthropic's public API
    from anthropic_api import (
        query_llm,
        query_llm_with_usage,
        query_with_retry,
        AVAILABLE_MODELS,
        DEFAULT_MODEL,
    )
    print("   → Anthropic API (public)")
    print(f"   → Default model: {DEFAULT_MODEL}")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("   ⚠️  Warning: ANTHROPIC_API_KEY not set!")

elif provider == "openai":
    # Placeholder for OpenAI implementation
    raise NotImplementedError(
        "OpenAI provider not yet implemented. "
        "Use 'apple' or 'anthropic' for now."
    )

else:
    # Default to Apple's internal API
    from token_counter import (
        query_llm,
        query_llm_with_usage,
        query_with_retry,
    )
    print("   → Apple internal API")
    print("   → Default model: openai/aws:anthropic.claude-sonnet-4-20250514-v1:0")

# Export the functions so they can be imported from this module
__all__ = [
    "query_llm",
    "query_llm_with_usage",
    "query_with_retry",
]

# Optional: Add provider info for debugging
CURRENT_PROVIDER = provider


def get_provider_info():
    """Get information about the current provider."""
    return {
        "provider": CURRENT_PROVIDER,
        "env_var": os.getenv("LLM_PROVIDER", "not set (defaulting to apple)"),
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


if __name__ == "__main__":
    # Test the router when run directly
    print("\n" + "="*70)
    print("LLM API Router Test")
    print("="*70)

    info = get_provider_info()
    print(f"Current provider: {info['provider']}")
    print(f"LLM_PROVIDER env: {info['env_var']}")
    print(f"ANTHROPIC_API_KEY set: {info['anthropic_key_set']}")

    print("\nTo switch providers:")
    print("  export LLM_PROVIDER=anthropic")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")

    print("\nImported functions:")
    print(f"  - query_llm: {'✓' if 'query_llm' in dir() else '✗'}")
    print(f"  - query_llm_with_usage: {'✓' if 'query_llm_with_usage' in dir() else '✗'}")
    print(f"  - query_with_retry: {'✓' if 'query_with_retry' in dir() else '✗'}")

    # Try a test query if possible
    print("\nTesting API...")
    try:
        result = query_llm_with_usage(
            "What is 1+1? Answer in exactly one word.",
            max_tokens=10
        )
        print(f"✅ API test successful!")
        print(f"   Response: {result['response'][:50]}...")
        print(f"   Tokens: {result.get('total_tokens', 'N/A')}")
    except Exception as e:
        print(f"❌ API test failed: {str(e)[:100]}")