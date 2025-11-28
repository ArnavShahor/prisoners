#!/usr/bin/env python3
"""
Prisoner's Dilemma Game with LLM Agents
Plays 100 rounds between two AI agents with token tracking.
"""

import json
import random
import time
from typing import Any
from token_counter import query_llm_with_usage


# ============================================================================
# GAME CONFIGURATION
# ============================================================================

# Payoff matrix: (Your score, Opponent score)
# First index: your move (C=0, D=1)
# Second index: opponent's move (C=0, D=1)
PAYOFF_MATRIX = {
    ("C", "C"): (3, 3),  # Both cooperate
    ("C", "D"): (0, 5),  # You cooperate, opponent defects
    ("D", "C"): (5, 0),  # You defect, opponent cooperates
    ("D", "D"): (1, 1),  # Both defect
}

ACTUAL_ROUNDS = 100
ANNOUNCED_MIN_ROUNDS = 85
ANNOUNCED_MAX_ROUNDS = 115


# ============================================================================
# AGENT CLASS
# ============================================================================


class PrisonersDilemmaAgent:
    """
    AI agent that plays Prisoner's Dilemma using LLM reasoning.
    """

    def __init__(
        self,
        name: str,
        strategy_prompt: str = "",
        model: str = "openai/aws:anthropic.claude-sonnet-4-5-20250929-v1:0",
    ):
        """
        Initialize agent.

        Args:
            name: Agent's name
            strategy_prompt: Additional instructions for strategy (optional)
            model: LLM model to use
        """
        self.name = name
        self.strategy_prompt = strategy_prompt
        self.model = model
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.query_count = 0
        self.total_time = 0.0  # Track total time for all queries

    def decide(
        self, round_num: int, history: list[dict[str, Any]], max_rounds_announced: str
    ) -> tuple[str, dict[str, Any]]:
        """
        Make a decision for this round.

        Args:
            round_num: Current round number
            history: List of previous rounds
            max_rounds_announced: String describing announced game length

        Returns:
            Tuple of (decision, usage_info)
            decision: "C" for cooperate or "D" for defect
        """
        # Build prompt
        prompt = self._build_prompt(round_num, history, max_rounds_announced)

        # Query LLM with usage tracking and timing
        start_time = time.time()
        result = query_llm_with_usage(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            model=self.model,
            max_tokens=512,
        )
        elapsed_time = time.time() - start_time

        # Update token tracking
        self.total_prompt_tokens += result["prompt_tokens"]
        self.total_completion_tokens += result["completion_tokens"]
        self.total_tokens += result["total_tokens"]
        self.query_count += 1
        self.total_time += elapsed_time

        # Extract decision
        decision = self._extract_decision(result["response"])

        return decision, result

    def _get_system_prompt(self) -> str:
        """Build system prompt for the agent."""
        base_prompt = """You are playing the Prisoner's Dilemma game.

PAYOFF MATRIX (your score, opponent score):
- Both Cooperate (C, C): (3, 3)
- You Cooperate, They Defect (C, D): (0, 5)
- You Defect, They Cooperate (D, C): (5, 0)
- Both Defect (D, D): (1, 1)

Your goal is to maximize YOUR total score across all rounds.

You must respond with ONLY a JSON object in this format:
{
  "reasoning": "brief explanation of your thinking",
  "decision": "C"
}

where decision is either "C" (cooperate) or "D" (defect)."""

        if self.strategy_prompt:
            base_prompt += f"\n\nADDITIONAL STRATEGY: {self.strategy_prompt}"

        return base_prompt

    def _build_prompt(
        self, round_num: int, history: list[dict[str, Any]], max_rounds_announced: str
    ) -> str:
        """Build the decision prompt."""
        prompt = f"""This is round {round_num}.
Game length: {max_rounds_announced}

"""

        if history:
            prompt += "GAME HISTORY:\n"
            for i, h in enumerate(history, 1):
                prompt += f"Round {i}: You played {h['your_move']}, Opponent played {h['opp_move']} → You scored {h['your_score']}, Opponent scored {h['opp_score']}\n"

            # Add score summary
            your_total = sum(h["your_score"] for h in history)
            opp_total = sum(h["opp_score"] for h in history)
            prompt += f"\nCurrent totals: You={your_total}, Opponent={opp_total}\n"

        prompt += "\nWhat is your decision for this round? Respond with JSON only."

        return prompt

    def _extract_decision(self, response: str) -> str:
        """Extract decision from LLM response."""
        # Try to parse JSON
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]

            data = json.loads(response)
            decision = data.get("decision", "").upper()

            if decision in ["C", "D"]:
                return decision
            else:
                print(f"Invalid decision '{decision}', defaulting to D")
                return "D"

        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response was: {response}")
            # Default to defect if parsing fails
            return "D"

    def get_stats(self) -> dict[str, Any]:
        """Get token usage statistics."""
        return {
            "name": self.name,
            "total_queries": self.query_count,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "avg_tokens_per_query": (
                self.total_tokens / self.query_count if self.query_count > 0 else 0
            ),
            "total_time": self.total_time,
            "avg_time_per_query": (
                self.total_time / self.query_count if self.query_count > 0 else 0
            ),
        }


# ============================================================================
# GAME ENGINE
# ============================================================================


class PrisonersDilemmaGame:
    """
    Game engine that runs the Prisoner's Dilemma between two agents.
    """

    def __init__(self, agent1: PrisonersDilemmaAgent, agent2: PrisonersDilemmaAgent):
        """Initialize game with two agents."""
        self.agent1 = agent1
        self.agent2 = agent2
        self.history = []  # Full game history
        self.agent1_history = []  # Agent 1's perspective
        self.agent2_history = []  # Agent 2's perspective

    def play_round(self, round_num: int) -> dict[str, Any]:
        """
        Play a single round.

        Returns:
            Dictionary with round results
        """
        max_rounds_msg = (
            f"a random number between {ANNOUNCED_MIN_ROUNDS} and {ANNOUNCED_MAX_ROUNDS}"
        )

        # Both agents decide
        decision1, usage1 = self.agent1.decide(
            round_num, self.agent1_history, max_rounds_msg
        )
        decision2, usage2 = self.agent2.decide(
            round_num, self.agent2_history, max_rounds_msg
        )

        # Calculate payoffs
        score1, score2 = PAYOFF_MATRIX[(decision1, decision2)]

        # Record in histories
        round_data = {
            "round": round_num,
            "agent1_move": decision1,
            "agent2_move": decision2,
            "agent1_score": score1,
            "agent2_score": score2,
        }

        self.history.append(round_data)

        # Update agent-specific histories
        self.agent1_history.append(
            {
                "your_move": decision1,
                "opp_move": decision2,
                "your_score": score1,
                "opp_score": score2,
            }
        )

        self.agent2_history.append(
            {
                "your_move": decision2,
                "opp_move": decision1,
                "your_score": score2,
                "opp_score": score1,
            }
        )

        return round_data

    def play_game(self, rounds: int = ACTUAL_ROUNDS, verbose: bool = True) -> dict[str, Any]:
        """
        Play the full game.

        Args:
            rounds: Number of rounds to play
            verbose: Print progress

        Returns:
            Game results dictionary
        """
        if verbose:
            print("=" * 70)
            print("PRISONER'S DILEMMA GAME")
            print("=" * 70)
            print(f"Agent 1: {self.agent1.name}")
            print(f"Agent 2: {self.agent2.name}")
            print(f"Announced rounds: {ANNOUNCED_MIN_ROUNDS}-{ANNOUNCED_MAX_ROUNDS}")
            print(f"Actual rounds: {rounds}")
            print("=" * 70)
            print()

        game_start_time = time.time()

        for round_num in range(1, rounds + 1):
            if verbose:
                print(f"Playing round {round_num}/{rounds}...")

            self.play_round(round_num)

        game_total_time = time.time() - game_start_time

        # Calculate final results
        results = self._calculate_results()
        results["timing"] = {
            "total_game_time": game_total_time,
            "avg_round_time": game_total_time / rounds,
        }

        if verbose:
            self._print_results(results)

        return results

    def _calculate_results(self) -> dict[str, Any]:
        """Calculate final game results."""
        agent1_total = sum(h["agent1_score"] for h in self.history)
        agent2_total = sum(h["agent2_score"] for h in self.history)

        # Count cooperation/defection
        agent1_cooperations = sum(1 for h in self.history if h["agent1_move"] == "C")
        agent2_cooperations = sum(1 for h in self.history if h["agent2_move"] == "C")

        # Get token stats
        agent1_stats = self.agent1.get_stats()
        agent2_stats = self.agent2.get_stats()

        return {
            "rounds_played": len(self.history),
            "agent1": {
                "name": self.agent1.name,
                "total_score": agent1_total,
                "cooperations": agent1_cooperations,
                "defections": len(self.history) - agent1_cooperations,
                "tokens": agent1_stats,
            },
            "agent2": {
                "name": self.agent2.name,
                "total_score": agent2_total,
                "cooperations": agent2_cooperations,
                "defections": len(self.history) - agent2_cooperations,
                "tokens": agent2_stats,
            },
            "winner": (
                self.agent1.name
                if agent1_total > agent2_total
                else (
                    self.agent2.name
                    if agent2_total > agent1_total
                    else "TIE"
                )
            ),
            "history": self.history,
        }

    def _print_results(self, results: dict[str, Any]):
        """Print game results."""
        print("\n" + "=" * 70)
        print("GAME RESULTS")
        print("=" * 70)

        for agent_key in ["agent1", "agent2"]:
            agent = results[agent_key]
            print(f"\n{agent['name']}:")
            print(f"  Total Score: {agent['total_score']}")
            print(
                f"  Cooperations: {agent['cooperations']} ({agent['cooperations']/results['rounds_played']*100:.1f}%)"
            )
            print(
                f"  Defections: {agent['defections']} ({agent['defections']/results['rounds_played']*100:.1f}%)"
            )

        print(f"\nWINNER: {results['winner']}")

        print("\n" + "=" * 70)
        print("TOKEN USAGE")
        print("=" * 70)

        total_tokens = 0
        for agent_key in ["agent1", "agent2"]:
            agent = results[agent_key]
            tokens = agent["tokens"]
            total_tokens += tokens["total_tokens"]
            print(f"\n{agent['name']}:")
            print(f"  Total queries: {tokens['total_queries']}")
            print(f"  Total tokens: {tokens['total_tokens']:,}")
            print(f"  Prompt tokens: {tokens['prompt_tokens']:,}")
            print(f"  Completion tokens: {tokens['completion_tokens']:,}")
            print(f"  Avg tokens/query: {tokens['avg_tokens_per_query']:.1f}")

        print(f"\nCOMBINED TOTAL: {total_tokens:,} tokens")

        # Calculate cost (Claude Sonnet 4.5 pricing)
        total_input = sum(
            results[f"agent{i+1}"]["tokens"]["prompt_tokens"] for i in range(2)
        )
        total_output = sum(
            results[f"agent{i+1}"]["tokens"]["completion_tokens"] for i in range(2)
        )

        input_cost = (total_input / 1_000_000) * 3.00
        output_cost = (total_output / 1_000_000) * 15.00
        total_cost = input_cost + output_cost

        print(f"\nESTIMATED COST (Claude Sonnet 4.5):")
        print(f"  Input cost:  ${input_cost:.6f}")
        print(f"  Output cost: ${output_cost:.6f}")
        print(f"  Total cost:  ${total_cost:.6f}")

        # Print timing information
        print("\n" + "=" * 70)
        print("TIMING")
        print("=" * 70)

        for agent_key in ["agent1", "agent2"]:
            agent = results[agent_key]
            tokens = agent["tokens"]
            print(f"\n{agent['name']}:")
            print(f"  Total time: {tokens['total_time']:.2f}s")
            print(f"  Avg time/query: {tokens['avg_time_per_query']:.2f}s")

        print(f"\nGame totals:")
        print(f"  Total game time: {results['timing']['total_game_time']:.2f}s")
        print(f"  Avg round time: {results['timing']['avg_round_time']:.2f}s")

        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run the Prisoner's Dilemma game."""
    # Create two agents
    agent1 = PrisonersDilemmaAgent(
        name="Agent Alpha",
        strategy_prompt="Try to maximize long-term cooperation for mutual benefit.",
    )

    agent2 = PrisonersDilemmaAgent(
        name="Agent Beta",
        strategy_prompt="Be strategic and adaptive based on opponent's behavior.",
    )

    # Create and play game
    game = PrisonersDilemmaGame(agent1, agent2)
    results = game.play_game(rounds=ACTUAL_ROUNDS, verbose=True)

    # Save results to file
    output_file = "game_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
