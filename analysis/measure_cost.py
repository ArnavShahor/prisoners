#!/usr/bin/env python3
"""
Quick script to measure the cost of a specific message.
"""

import os
from anthropic import Anthropic
from anthropic.types import TextBlock

# Anthropic pricing (as of 2025)
# Claude Sonnet 4.5: $3 per million input tokens, $15 per million output tokens
PRICE_PER_MILLION_INPUT = 3.00
PRICE_PER_MILLION_OUTPUT = 15.00

def calculate_cost(input_tokens: int, output_tokens: int) -> dict:
    """Calculate the cost in USD for given token counts."""
    input_cost = (input_tokens / 1_000_000) * PRICE_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * PRICE_PER_MILLION_OUTPUT
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def measure_message_cost(message: str):
    """Send a message to Claude and measure its cost."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return

    client = Anthropic(api_key=api_key)

    print(f"Measuring cost for message: \"{message}\"\n")
    print("Sending to Claude API...\n")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": message}
            ]
        )

        # Extract response text
        response_text = ""
        for block in response.content:
            if isinstance(block, TextBlock):
                response_text += block.text

        # Get token counts
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate costs
        costs = calculate_cost(input_tokens, output_tokens)

        # Display results
        print("=" * 60)
        print("TOKEN USAGE")
        print("=" * 60)
        print(f"Input tokens:  {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print(f"Total tokens:  {total_tokens:,}")
        print()

        print("=" * 60)
        print("COST BREAKDOWN (USD)")
        print("=" * 60)
        print(f"Input cost:  ${costs['input_cost']:.6f}")
        print(f"Output cost: ${costs['output_cost']:.6f}")
        print(f"Total cost:  ${costs['total_cost']:.6f}")
        print()

        # Show cost per 1000 calls
        cost_per_1000 = costs['total_cost'] * 1000
        print(f"Cost for 1,000 calls: ${cost_per_1000:.2f}")
        print(f"Cost for 10,000 calls: ${cost_per_1000 * 10:.2f}")
        print()

        print("=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(response_text)
        print()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Measure the specific message
    measure_message_cost("play the new round")
