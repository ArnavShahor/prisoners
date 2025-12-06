#!/usr/bin/env python3
"""
Ultimatum Game with LLM Agents
One-shot ultimatum games between AI agents with different personalities.
"""

import argparse
import json
import time
from typing import Any
from llm_api import query_llm_with_usage


# ============================================================================
# GAME CONFIGURATION
# ============================================================================

# Default configuration values (can be overridden via command-line arguments)
DEFAULT_TOTAL_AMOUNT = 100  # Points to split
DEFAULT_PLAYER_INDICES = [0, 1, 2, 3, 4, 5]  # Players 1-6
DEFAULT_GAMES_PER_DIRECTION = 5  # Number of games each agent plays as proposer
DEFAULT_PROPOSER_ONLY_MODE = True  # True = only proposer offers (fast), False = full game
DEFAULT_SELF_DESCRIPTION = "minimal"  # "full", "limited", or "minimal"
DEFAULT_OPPONENT_DESCRIPTION = "minimal"  # "full", "limited", or "minimal"
DEFAULT_PERSONAS_FILE = "Personas_Jobs.json"
DEFAULT_TRANSFER_RATE = 1.0  # Transfer efficiency (1.0 = no loss, 0.5 = 50% loss)

# Runtime configuration (set by command-line args or defaults)
TOTAL_AMOUNT = DEFAULT_TOTAL_AMOUNT
PROPOSER_ONLY_MODE = DEFAULT_PROPOSER_ONLY_MODE
SELF_DESCRIPTION = DEFAULT_SELF_DESCRIPTION
OPPONENT_DESCRIPTION = DEFAULT_OPPONENT_DESCRIPTION
TRANSFER_RATE = DEFAULT_TRANSFER_RATE


# ============================================================================
# AGENT CLASS
# ============================================================================


class UltimatumAgent:
    """
    AI agent that plays Ultimatum Game using LLM reasoning.
    Can play both proposer and responder roles.
    """

    def __init__(
        self,
        persona: dict[str, Any],
        model: str = None,  # Use provider's default model
    ):
        """
        Initialize agent with a personality.

        Args:
            persona: Dictionary with personality data from Personas_Jobs.json
            model: LLM model to use
        """
        self.persona = persona
        self.name = persona["name"]
        self.model = model

        # Token tracking
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.query_count = 0
        self.failed_queries = 0  # Track failed API calls
        self.total_time = 0.0

    def make_proposal(
        self, responder_persona: dict[str, Any]
    ) -> tuple[int, int, dict[str, Any]]:
        """
        Make a proposal as the proposer.

        Args:
            responder_persona: Full personality dict of the responder

        Returns:
            Tuple of (keep_amount, actual_offer, usage_info)
            - keep_amount: Amount proposer wants to keep
            - actual_offer: Amount responder receives (after transfer rate)
            - usage_info: Dictionary with token usage, reasoning, etc.
        """
        # Build prompt
        prompt = self._build_proposal_prompt(responder_persona)

        # Query LLM with usage tracking and timing
        start_time = time.time()
        # Build kwargs for the API call
        api_kwargs = {
            "prompt": prompt,
            "system_prompt": self._get_proposal_system_prompt(),
            "max_tokens": 512,
        }
        # Only add model if it's not None
        if self.model is not None:
            api_kwargs["model"] = self.model

        result = query_llm_with_usage(**api_kwargs)
        elapsed_time = time.time() - start_time

        # Update token tracking
        self.total_prompt_tokens += result["prompt_tokens"]
        self.total_completion_tokens += result["completion_tokens"]
        self.total_tokens += result["total_tokens"]
        self.query_count += 1
        self.total_time += elapsed_time

        # Track failures
        if result.get("failed", False):
            self.failed_queries += 1

        # Extract keep amount and calculate actual offer
        keep_amount, actual_offer, reasoning = self._extract_offer(result["response"])

        # Add reasoning, keep_amount, and actual_offer to result
        result["reasoning"] = reasoning
        result["keep_amount"] = keep_amount
        result["actual_offer"] = actual_offer

        return keep_amount, actual_offer, result

    def respond_to_offer(
        self, proposer_persona: dict[str, Any], keep_amount: int, offer: int
    ) -> tuple[str, dict[str, Any]]:
        """
        Respond to an offer as the responder.

        Args:
            proposer_persona: Full personality dict of the proposer
            keep_amount: Amount proposer is keeping
            offer: Amount offered to this agent (after transfer rate)

        Returns:
            Tuple of ("accept" or "reject", usage_info)
        """
        # Build prompt
        prompt = self._build_response_prompt(proposer_persona, keep_amount, offer)

        # Query LLM with usage tracking and timing
        start_time = time.time()
        # Build kwargs for the API call
        api_kwargs = {
            "prompt": prompt,
            "system_prompt": self._get_response_system_prompt(),
            "max_tokens": 512,
        }
        # Only add model if it's not None
        if self.model is not None:
            api_kwargs["model"] = self.model

        result = query_llm_with_usage(**api_kwargs)
        elapsed_time = time.time() - start_time

        # Update token tracking
        self.total_prompt_tokens += result["prompt_tokens"]
        self.total_completion_tokens += result["completion_tokens"]
        self.total_tokens += result["total_tokens"]
        self.query_count += 1
        self.total_time += elapsed_time

        # Track failures
        if result.get("failed", False):
            self.failed_queries += 1

        # Extract decision
        decision, reasoning = self._extract_decision(result["response"])

        # Add reasoning to result
        result["reasoning"] = reasoning

        return decision, result

    def _get_proposal_system_prompt(self) -> str:
        """Build system prompt for proposer role."""
        transfer_info = ""
        if TRANSFER_RATE != 1.0:
            if TRANSFER_RATE < 1.0:
                loss_pct = (1.0 - TRANSFER_RATE) * 100
                transfer_info = f"\n\nIMPORTANT: There is a transfer inefficiency of {loss_pct:.0f}%. The other person receives only {TRANSFER_RATE:.0%} of what you give up."
            else:
                gain_pct = (TRANSFER_RATE - 1.0) * 100
                transfer_info = f"\n\nIMPORTANT: There is a transfer amplification of {gain_pct:.0f}%. The other person receives {TRANSFER_RATE:.0%} of what you give up (more than you lose)."

        return f"""You are dividing $100 with a stranger you have never met and will never meet again.

This is a one-time, anonymous interaction.

RULES:
- You decide how much to KEEP for yourself ($0-$100)
- The other person receives: ($100 - keep) × {TRANSFER_RATE}
- They can accept or reject your proposal
- If accepted: you get your 'keep' amount, they get ($100 - keep) × {TRANSFER_RATE}
- If rejected: both receive $0{transfer_info}

Your goal is to maximize your money.

You must respond with ONLY a JSON object in this format:
{{
  "reasoning": "brief explanation of your strategic thinking",
  "keep": <number between 0 and 100>
}}

where keep is the dollar amount you want to keep for yourself."""

    def _get_response_system_prompt(self) -> str:
        """Build system prompt for responder role."""
        return """A stranger you have never met and will never meet again has proposed how to split $100 with you.

This is a one-time, anonymous interaction.

RULES:
- You can either ACCEPT or REJECT their proposal
- If you accept: you both receive the proposed amounts
- If you reject: you both receive $0

Decide what feels right to you given the situation.

You must respond with ONLY a JSON object in this format:
{
  "reasoning": "brief explanation of your thinking",
  "decision": "accept" or "reject"
}

where decision is either "accept" or "reject"."""

    def _build_proposal_prompt(self, responder_persona: dict[str, Any]) -> str:
        """Build the proposal prompt."""
        # Build self description based on setting
        if SELF_DESCRIPTION == "full":
            self_info = f"""You are {self.persona['name']}, age {self.persona['age']}, works as {self.persona['job']}.
Your description: {self.persona['description']}"""
        elif SELF_DESCRIPTION == "limited":
            self_info = f"""You are {self.persona['name']}, age {self.persona['age']}, works as {self.persona['job']}, from {self.persona['location']}."""
        else:  # minimal
            first_name = self.persona['name'].split()[0]
            self_info = f"""You are {first_name}, {self.persona['job']}."""

        # Build opponent description based on setting
        if OPPONENT_DESCRIPTION == "full":
            opponent_info = f"""About {responder_persona['name']}:
- Gender: {responder_persona['gender']}
- Age: {responder_persona['age']}
- Job: {responder_persona['job']}
- Location: {responder_persona['location']}
- Description: {responder_persona['description']}"""
        elif OPPONENT_DESCRIPTION == "limited":
            opponent_info = f"""About {responder_persona['name']}:
- Gender: {responder_persona['gender']}
- Age: {responder_persona['age']}
- Job: {responder_persona['job']}
- Location: {responder_persona['location']}"""
        else:  # minimal
            first_name = responder_persona['name'].split()[0]
            opponent_info = f"""About {first_name}: {responder_persona['job']}"""

        # Use first names for minimal mode in the prompt text
        if OPPONENT_DESCRIPTION == "minimal":
            opponent_first_name = responder_persona['name'].split()[0]
        else:
            opponent_first_name = responder_persona['name']

        prompt = f"""{self_info}

You are dividing money with {opponent_first_name}.
{opponent_info}

You must split $100 between yourself and {opponent_first_name}.
Decide how much to offer {opponent_first_name}.

Respond with JSON only."""

        return prompt

    def _build_response_prompt(
        self, proposer_persona: dict[str, Any], keep_amount: int, offer: int
    ) -> str:
        """Build the response prompt.

        Args:
            proposer_persona: Proposer's personality data
            keep_amount: Amount proposer is keeping
            offer: Amount responder will receive (after transfer rate)
        """
        # Build self description based on setting
        if SELF_DESCRIPTION == "full":
            self_info = f"""You are {self.persona['name']}, age {self.persona['age']}, works as {self.persona['job']}.
Your description: {self.persona['description']}"""
        elif SELF_DESCRIPTION == "limited":
            self_info = f"""You are {self.persona['name']}, age {self.persona['age']}, works as {self.persona['job']}, from {self.persona['location']}."""
        else:  # minimal
            first_name = self.persona['name'].split()[0]
            self_info = f"""You are {first_name}, {self.persona['job']}."""

        # Build opponent description based on setting
        if OPPONENT_DESCRIPTION == "full":
            opponent_info = f"""About {proposer_persona['name']}:
- Gender: {proposer_persona['gender']}
- Age: {proposer_persona['age']}
- Job: {proposer_persona['job']}
- Location: {proposer_persona['location']}
- Description: {proposer_persona['description']}"""
        elif OPPONENT_DESCRIPTION == "limited":
            opponent_info = f"""About {proposer_persona['name']}:
- Gender: {proposer_persona['gender']}
- Age: {proposer_persona['age']}
- Job: {proposer_persona['job']}
- Location: {proposer_persona['location']}"""
        else:  # minimal
            first_name = proposer_persona['name'].split()[0]
            opponent_info = f"""About {first_name}: {proposer_persona['job']}"""

        # Use first names for minimal mode in the prompt text
        if OPPONENT_DESCRIPTION == "minimal":
            opponent_first_name = proposer_persona['name'].split()[0]
        else:
            opponent_first_name = proposer_persona['name']

        # Build transfer rate explanation
        transfer_explanation = ""
        if TRANSFER_RATE != 1.0:
            gave_up = TOTAL_AMOUNT - keep_amount
            if TRANSFER_RATE < 1.0:
                loss = gave_up - offer
                transfer_explanation = f"\nNote: {opponent_first_name} gave up ${gave_up}, but due to {TRANSFER_RATE:.0%} transfer efficiency, you receive ${offer} (${loss:.0f} was lost in transfer)."
            else:
                gain = offer - gave_up
                transfer_explanation = f"\nNote: {opponent_first_name} gave up ${gave_up}, but due to {TRANSFER_RATE:.0%} transfer rate, you receive ${offer} (${gain:.0f} gain from amplification)."

        prompt = f"""{self_info}

{opponent_first_name} is dividing money with you.
{opponent_info}

SITUATION:
{opponent_first_name} decided to keep ${keep_amount} for themselves.
You receive: ${offer} (transfer rate: {TRANSFER_RATE}){transfer_explanation}

If you ACCEPT: you get ${offer}, {opponent_first_name} gets ${keep_amount}
If you REJECT: both of you get $0

Decide whether to accept or reject.

Respond with JSON only."""

        return prompt

    def _extract_offer(self, response: str) -> tuple[int, int, str]:
        """Extract keep amount from LLM response and calculate actual offer.

        Returns:
            Tuple of (keep_amount, actual_offer, reasoning)
            - keep_amount: Amount proposer wants to keep
            - actual_offer: Amount responder receives = (100 - keep) * TRANSFER_RATE
            - reasoning: Proposer's reasoning
        """
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]

            # Try to clean up common JSON issues
            response = response.strip()

            # Remove any trailing incomplete text after the closing brace
            if '{' in response and '}' in response:
                # Find the last complete JSON object
                last_brace = response.rfind('}')
                if last_brace != -1:
                    response = response[:last_brace + 1]

            data = json.loads(response.strip())

            # Try to get "keep" first, fallback to "offer" for backward compatibility
            if "keep" in data:
                keep_amount = int(data["keep"])
            elif "offer" in data:
                # If old format with "offer", convert: they offered X, so they keep 100-X
                offer_amount = int(data["offer"])
                keep_amount = TOTAL_AMOUNT - offer_amount
            else:
                keep_amount = 50  # Default

            reasoning = data.get("reasoning", "")

            # Validate keep is in range
            if 0 <= keep_amount <= TOTAL_AMOUNT:
                # Calculate actual offer with transfer rate
                actual_offer = (TOTAL_AMOUNT - keep_amount) * TRANSFER_RATE
                return keep_amount, int(actual_offer), reasoning
            else:
                print(f"⚠️  Keep amount {keep_amount} out of range, clamping to 0-{TOTAL_AMOUNT}")
                keep_amount = max(0, min(TOTAL_AMOUNT, keep_amount))
                actual_offer = (TOTAL_AMOUNT - keep_amount) * TRANSFER_RATE
                return keep_amount, int(actual_offer), reasoning

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {str(e)[:100]}")
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'\b\d+\b', response)
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num <= TOTAL_AMOUNT:
                    actual_offer = (TOTAL_AMOUNT - num) * TRANSFER_RATE
                    return num, int(actual_offer), f"Recovered from malformed JSON - extracted {num}"
            # Default to fair split if we can't extract anything
            keep_amount = 50
            actual_offer = (TOTAL_AMOUNT - keep_amount) * TRANSFER_RATE
            return keep_amount, int(actual_offer), "Recovered from malformed JSON - defaulted to 50"
        except Exception as e:
            print(f"⚠️  Unexpected error parsing offer: {str(e)[:100]}")
            keep_amount = 50
            actual_offer = (TOTAL_AMOUNT - keep_amount) * TRANSFER_RATE
            return keep_amount, int(actual_offer), f"Error: {str(e)[:100]} - defaulted to 50"

    def _extract_decision(self, response: str) -> tuple[str, str]:
        """Extract decision from LLM response."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]

            # Try to clean up common JSON issues
            response = response.strip()

            # Remove any trailing incomplete text after the closing brace
            if '{' in response and '}' in response:
                # Find the last complete JSON object
                last_brace = response.rfind('}')
                if last_brace != -1:
                    response = response[:last_brace + 1]

            data = json.loads(response.strip())
            decision = data.get("decision", "").lower()
            reasoning = data.get("reasoning", "")

            if decision in ["accept", "reject"]:
                return decision, reasoning
            else:
                # If decision field is invalid, look for accept/reject in reasoning or raw response
                combined_text = (reasoning + " " + response).lower()
                if "accept" in combined_text and "reject" not in combined_text:
                    return "accept", reasoning if reasoning else "Defaulted to accept based on text analysis"
                elif "reject" in combined_text:
                    return "reject", reasoning if reasoning else "Defaulted to reject based on text analysis"
                else:
                    print(f"⚠️  Invalid decision '{decision}', defaulting to accept")
                    return "accept", reasoning if reasoning else "Invalid decision format"

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {str(e)[:100]}")
            # Try to extract decision from malformed response
            response_lower = response.lower()
            if "reject" in response_lower:
                return "reject", "Recovered from malformed JSON - found 'reject' in text"
            else:
                # Default to accept if we can't parse
                return "accept", "Recovered from malformed JSON - defaulted to accept"
        except Exception as e:
            print(f"⚠️  Unexpected error parsing response: {str(e)[:100]}")
            return "accept", f"Error: {str(e)[:100]} - defaulted to accept"

    def get_stats(self) -> dict[str, Any]:
        """Get token usage statistics."""
        return {
            "name": self.name,
            "total_queries": self.query_count,
            "failed_queries": self.failed_queries,
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


class UltimatumGame:
    """
    Game engine that runs a single Ultimatum Game between two agents.
    """

    def __init__(
        self, proposer: UltimatumAgent, responder: UltimatumAgent, game_id: int = 1
    ):
        """Initialize game with two agents and role assignments."""
        self.proposer = proposer
        self.responder = responder
        self.game_id = game_id
        self.result: dict[str, Any] = {}

    def play(self, verbose: bool = True) -> dict[str, Any]:
        """
        Play a single game.

        Args:
            verbose: Print game progress

        Returns:
            Dictionary with game results
        """
        if PROPOSER_ONLY_MODE:
            return self._play_proposer_only(verbose)
        else:
            return self._play_full_game(verbose)

    def _play_proposer_only(self, verbose: bool = True) -> dict[str, Any]:
        """
        Play game in proposer-only mode (no responder decision).

        Args:
            verbose: Print game progress

        Returns:
            Dictionary with game results
        """
        if verbose:
            print(
                f"\nGame {self.game_id}: {self.proposer.name} → {self.responder.name} (offer only, no response)"
            )

        # Proposer makes offer
        keep_amount, actual_offer, proposal_usage = self.proposer.make_proposal(self.responder.persona)

        if verbose:
            print(f"  {self.proposer.name}'s reasoning: {proposal_usage['reasoning'][:80]}...")
            print(f"  Keep: ${keep_amount}, Offer: ${actual_offer} (rate: {TRANSFER_RATE})")

        # Calculate money lost/gained in transfer
        gave_up = TOTAL_AMOUNT - keep_amount
        money_lost = gave_up - actual_offer  # Negative if transfer rate > 1.0

        # Build result dictionary (proposer-only mode)
        self.result = {
            "game_id": self.game_id,
            "proposer_name": self.proposer.name,
            "responder_name": self.responder.name,
            "proposer_idx": self.proposer.persona["player_number"],
            "responder_idx": self.responder.persona["player_number"],
            "keep_amount": keep_amount,
            "offer": actual_offer,  # Amount responder would receive
            "transfer_rate": TRANSFER_RATE,
            "money_lost": money_lost,  # Money lost/gained in transfer
            "decision": "no_response",
            "proposer_payoff": 0,  # No payoffs in proposer-only mode
            "responder_payoff": 0,
            "proposer_reasoning": proposal_usage["reasoning"],
            "responder_reasoning": "",  # No responder in this mode
            "proposer_tokens": proposal_usage["total_tokens"],
            "responder_tokens": 0,  # No responder query
            "total_tokens": proposal_usage["total_tokens"],
            "proposer_failed": proposal_usage.get("failed", False),  # Track API failure
            "responder_failed": False,  # No responder in this mode
            # Store persona traits for analysis (not shown to agents)
            "proposer_traits": self.proposer.persona["traits"],
            "responder_traits": self.responder.persona["traits"],
            "proposer_age": self.proposer.persona["age"],
            "responder_age": self.responder.persona["age"],
            "proposer_gender": self.proposer.persona["gender"],
            "responder_gender": self.responder.persona["gender"],
            "proposer_job": self.proposer.persona["job"],
            "responder_job": self.responder.persona["job"],
        }

        return self.result

    def _play_full_game(self, verbose: bool = True) -> dict[str, Any]:
        """
        Play full game with proposer and responder decisions.

        Args:
            verbose: Print game progress

        Returns:
            Dictionary with game results
        """
        if verbose:
            print(
                f"\nGame {self.game_id}: {self.proposer.name} (proposer) → {self.responder.name} (responder)"
            )

        # Proposer makes offer
        keep_amount, actual_offer, proposal_usage = self.proposer.make_proposal(self.responder.persona)

        if verbose:
            print(f"  {self.proposer.name}'s reasoning: {proposal_usage['reasoning'][:80]}...")
            print(f"  Keep: ${keep_amount}, Offer to responder: ${actual_offer} (rate: {TRANSFER_RATE})")

        # Responder decides
        decision, response_usage = self.responder.respond_to_offer(
            self.proposer.persona, keep_amount, actual_offer
        )

        if verbose:
            print(f"  {self.responder.name}'s reasoning: {response_usage['reasoning'][:80]}...")
            print(f"  Decision: {decision}")

        # Calculate payoffs
        if decision == "accept":
            proposer_payoff = keep_amount
            responder_payoff = actual_offer
        else:
            proposer_payoff = 0
            responder_payoff = 0

        # Calculate money lost/gained in transfer
        gave_up = TOTAL_AMOUNT - keep_amount
        money_lost = gave_up - actual_offer  # Negative if transfer rate > 1.0

        if verbose:
            print(
                f"  Payoffs: {self.proposer.name}=${proposer_payoff}, {self.responder.name}=${responder_payoff}"
            )

        # Build result dictionary
        self.result = {
            "game_id": self.game_id,
            "proposer_name": self.proposer.name,
            "responder_name": self.responder.name,
            "proposer_idx": self.proposer.persona["player_number"],
            "responder_idx": self.responder.persona["player_number"],
            "keep_amount": keep_amount,
            "offer": actual_offer,  # Amount responder receives
            "transfer_rate": TRANSFER_RATE,
            "money_lost": money_lost,  # Money lost/gained in transfer
            "decision": decision,
            "proposer_payoff": proposer_payoff,
            "responder_payoff": responder_payoff,
            "proposer_reasoning": proposal_usage["reasoning"],
            "responder_reasoning": response_usage["reasoning"],
            "proposer_tokens": proposal_usage["total_tokens"],
            "responder_tokens": response_usage["total_tokens"],
            "total_tokens": proposal_usage["total_tokens"]
            + response_usage["total_tokens"],
            "proposer_failed": proposal_usage.get("failed", False),  # Track API failures
            "responder_failed": response_usage.get("failed", False),
            # Store persona traits for analysis (not shown to agents)
            "proposer_traits": self.proposer.persona["traits"],
            "responder_traits": self.responder.persona["traits"],
            "proposer_age": self.proposer.persona["age"],
            "responder_age": self.responder.persona["age"],
            "proposer_gender": self.proposer.persona["gender"],
            "responder_gender": self.responder.persona["gender"],
            "proposer_job": self.proposer.persona["job"],
            "responder_job": self.responder.persona["job"],
        }

        return self.result


# ============================================================================
# SIMULATION
# ============================================================================


def load_personas(filepath: str = "Personas_Jobs.json") -> list[dict[str, Any]]:
    """Load persona data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        personas = json.load(f)
    return personas


def run_simulation(
    player_indices: list[int] = None,
    games_per_direction: int = None,
    personas_file: str = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run the ultimatum game simulation.

    Args:
        player_indices: List of player indices to use (default: [0,1,2,3,4,5])
        games_per_direction: Number of games each agent plays as proposer (default: 5)
        personas_file: Path to personas JSON file (default: Personas_Jobs.json)
        verbose: Print progress

    Returns:
        Simulation results dictionary
    """
    # Use defaults if not provided
    if player_indices is None:
        player_indices = DEFAULT_PLAYER_INDICES
    if games_per_direction is None:
        games_per_direction = DEFAULT_GAMES_PER_DIRECTION
    if personas_file is None:
        personas_file = DEFAULT_PERSONAS_FILE

    print("=" * 70)
    print("ULTIMATUM GAME SIMULATION")
    print("=" * 70)

    # Load personas
    personas = load_personas(personas_file)
    print(f"Loaded {len(personas)} personas from {personas_file}")

    # Select agents
    agent_indices = player_indices
    num_agents = len(agent_indices)
    total_games = num_agents * (num_agents - 1) * games_per_direction
    print(f"Running with {num_agents} agents → {total_games} total games ({games_per_direction} games per direction)")
    print(f"Transfer rate: {TRANSFER_RATE}")

    # Get selected personas
    selected_personas = [personas[i] for i in agent_indices]

    print("=" * 70)
    print("\nSelected agents:")
    for i, persona in enumerate(selected_personas, 1):
        print(f"  {i}. {persona['name']} - {persona['age']}-year-old {persona['job']}")
        print(f"     {persona['description'][:80]}...")

    print("\n" + "=" * 70)

    # Create agents
    agents = [UltimatumAgent(persona) for persona in selected_personas]

    # Run games - each agent proposes multiple times
    results = []
    game_count = 0
    total_expected_games = len(agents) * (len(agents) - 1) * games_per_direction

    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:  # Don't play against self
                # Play multiple games in this direction
                for _ in range(games_per_direction):
                    game_count += 1

                    # Show progress bar if not verbose
                    if not verbose:
                        percent = (game_count / total_expected_games) * 100
                        bar_length = 50
                        filled = int(bar_length * game_count / total_expected_games)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f'\rProgress: [{bar}] {percent:.1f}% ({game_count}/{total_expected_games})', end='', flush=True)

                    game = UltimatumGame(
                        proposer=agents[i], responder=agents[j], game_id=game_count
                    )
                    result = game.play(verbose=verbose)
                    results.append(result)

    # Print newline after progress bar
    if not verbose:
        print()  # Move to next line after progress bar

    # Calculate summary statistics
    total_games = len(results)
    accepted = sum(1 for r in results if r["decision"] == "accept")
    rejected = total_games - accepted
    avg_offer = sum(r["offer"] for r in results) / total_games if total_games > 0 else 0
    avg_proposer_payoff = (
        sum(r["proposer_payoff"] for r in results) / total_games
        if total_games > 0
        else 0
    )
    avg_responder_payoff = (
        sum(r["responder_payoff"] for r in results) / total_games
        if total_games > 0
        else 0
    )

    # Calculate total tokens and cost
    total_tokens = sum(r["total_tokens"] for r in results)
    total_input = sum(agent.total_prompt_tokens for agent in agents)
    total_output = sum(agent.total_completion_tokens for agent in agents)

    # Calculate failed API calls
    total_failed_calls = sum(agent.failed_queries for agent in agents)
    total_api_calls = sum(agent.query_count for agent in agents)
    failed_percentage = (total_failed_calls / total_api_calls * 100) if total_api_calls > 0 else 0

    # Claude Sonnet 4.5 pricing
    input_cost = (total_input / 1_000_000) * 3.00
    output_cost = (total_output / 1_000_000) * 15.00
    total_cost = input_cost + output_cost

    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Total games: {total_games}")
    print(f"Total accepted: {accepted} ({accepted/total_games*100:.1f}%)")
    print(f"Total rejected: {rejected} ({rejected/total_games*100:.1f}%)")
    print(f"Average offer: {avg_offer:.1f}")
    print(f"Average proposer payoff: {avg_proposer_payoff:.1f}")
    print(f"Average responder payoff: {avg_responder_payoff:.1f}")
    print()
    print(f"API Call Statistics:")
    print(f"  Total API calls: {total_api_calls:,}")
    print(f"  Failed calls: {total_failed_calls:,} ({failed_percentage:.1f}%)")
    print(f"  Successful calls: {total_api_calls - total_failed_calls:,} ({100 - failed_percentage:.1f}%)")
    print()
    print(f"Token Usage: {total_tokens:,} tokens")
    print(f"  Input tokens:  {total_input:,}")
    print(f"  Output tokens: {total_output:,}")
    print(f"Estimated Cost: ${total_cost:.4f}")
    print("=" * 70)

    return {
        "results": results,
        "summary": {
            "total_games": total_games,
            "accepted": accepted,
            "rejected": rejected,
            "avg_offer": avg_offer,
            "avg_proposer_payoff": avg_proposer_payoff,
            "avg_responder_payoff": avg_responder_payoff,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_api_calls": total_api_calls,
            "failed_calls": total_failed_calls,
            "failed_percentage": failed_percentage,
        },
        "agents": [agent.get_stats() for agent in agents],
    }


def save_results_to_csv(
    results: list[dict[str, Any]], filename: str = "ultimatum_results.csv"
):
    """Save results to CSV file."""
    import csv

    if not results:
        print("No results to save")
        return

    # Define CSV columns
    fieldnames = [
        "game_id",
        "proposer_name",
        "responder_name",
        "proposer_idx",
        "responder_idx",
        "offer",
        "decision",
        "proposer_payoff",
        "responder_payoff",
        "proposer_age",
        "proposer_gender",
        "proposer_job",
        "responder_age",
        "responder_gender",
        "responder_job",
        "euclidean_distance",
        "manhattan_distance",
        "proposer_tokens",
        "responder_tokens",
        "total_tokens",
        "proposer_failed",
        "responder_failed",
        "proposer_reasoning",
        "responder_reasoning",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Calculate distance metrics between proposer and responder traits
            p_traits = result["proposer_traits"]
            r_traits = result["responder_traits"]

            # Euclidean distance: sqrt(sum of squared differences)
            euclidean = (
                (p_traits["FS"] - r_traits["FS"]) ** 2 +
                (p_traits["GR"] - r_traits["GR"]) ** 2 +
                (p_traits["RA"] - r_traits["RA"]) ** 2 +
                (p_traits["SS"] - r_traits["SS"]) ** 2 +
                (p_traits["ST"] - r_traits["ST"]) ** 2
            ) ** 0.5

            # Manhattan distance: sum of absolute differences
            manhattan = (
                abs(p_traits["FS"] - r_traits["FS"]) +
                abs(p_traits["GR"] - r_traits["GR"]) +
                abs(p_traits["RA"] - r_traits["RA"]) +
                abs(p_traits["SS"] - r_traits["SS"]) +
                abs(p_traits["ST"] - r_traits["ST"])
            )

            row = {
                "game_id": result["game_id"],
                "proposer_name": result["proposer_name"],
                "responder_name": result["responder_name"],
                "proposer_idx": result["proposer_idx"],
                "responder_idx": result["responder_idx"],
                "offer": result["offer"],
                "decision": result["decision"],
                "proposer_payoff": result["proposer_payoff"],
                "responder_payoff": result["responder_payoff"],
                "proposer_age": result["proposer_age"],
                "proposer_gender": result["proposer_gender"],
                "proposer_job": result["proposer_job"],
                "responder_age": result["responder_age"],
                "responder_gender": result["responder_gender"],
                "responder_job": result["responder_job"],
                "euclidean_distance": round(euclidean, 4),
                "manhattan_distance": round(manhattan, 4),
                "proposer_tokens": result["proposer_tokens"],
                "responder_tokens": result["responder_tokens"],
                "total_tokens": result["total_tokens"],
                "proposer_failed": result["proposer_failed"],
                "responder_failed": result["responder_failed"],
                "proposer_reasoning": result["proposer_reasoning"],
                "responder_reasoning": result["responder_reasoning"],
            }
            writer.writerow(row)

    print(f"\nResults saved to {filename}")


# ============================================================================
# MAIN
# ============================================================================


def parse_arguments():
    """Parse command-line arguments for the Ultimatum Game simulation."""
    parser = argparse.ArgumentParser(
        description="Run Ultimatum Game simulation with AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with players 1-6, 5 games per direction
  python ultimatum_game.py --players 0-5 --games 5

  # Run with specific players, 10 games, verbose
  python ultimatum_game.py -p 0,1,2 -g 10 -v

  # Run full game mode with custom personas file
  python ultimatum_game.py --full-game --personas-file custom_personas.json

  # Run with full descriptions
  python ultimatum_game.py --self-description full --opponent-description full
        """
    )

    # Player configuration
    parser.add_argument(
        "-p", "--players",
        type=str,
        default="0,1,2,3,4,5",
        help="Player indices, comma-separated (e.g., '0,1,2') or range (e.g., '0-5'). Default: 0,1,2,3,4,5"
    )

    parser.add_argument(
        "-g", "--games",
        type=int,
        default=5,
        help="Number of games each agent plays as proposer. Default: 5"
    )

    # Game configuration
    parser.add_argument(
        "--total-amount",
        type=int,
        default=100,
        help="Total points to split in each game. Default: 100"
    )

    parser.add_argument(
        "--proposer-only",
        action="store_true",
        default=True,
        help="Only collect proposer offers (fast mode). Default: True"
    )

    parser.add_argument(
        "--full-game",
        action="store_true",
        default=False,
        help="Run full game with responder accept/reject decisions (slower)"
    )

    parser.add_argument(
        "--transfer-rate",
        type=float,
        default=1.0,
        help="Transfer rate (must be > 0, no upper limit). Responder receives (100-keep) × rate. 1.0=no loss, 0.5=50%% loss, 2.0=2x gain. Default: 1.0"
    )

    # Description settings
    parser.add_argument(
        "--self-description",
        type=str,
        choices=["full", "limited", "minimal"],
        default="minimal",
        help="Self description level. Default: minimal"
    )

    parser.add_argument(
        "--opponent-description",
        type=str,
        choices=["full", "limited", "minimal"],
        default="minimal",
        help="Opponent description level. Default: minimal"
    )

    # File configuration
    parser.add_argument(
        "--personas-file",
        type=str,
        default="Personas_Jobs.json",
        help="Path to personas JSON file. Default: Personas_Jobs.json"
    )

    # Output configuration
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print detailed progress during simulation"
    )

    args = parser.parse_args()

    # Parse player indices
    if "-" in args.players:
        # Handle range format (e.g., "0-5")
        start, end = map(int, args.players.split("-"))
        args.player_indices = list(range(start, end + 1))
    else:
        # Handle comma-separated format (e.g., "0,1,2")
        args.player_indices = [int(x.strip()) for x in args.players.split(",")]

    # Handle full-game flag (overrides proposer-only)
    if args.full_game:
        args.proposer_only = False
    else:
        args.proposer_only = True

    # Validate transfer rate (must be > 0, no upper bound)
    if args.transfer_rate <= 0:
        parser.error(f"transfer-rate must be greater than 0, got {args.transfer_rate}")

    return args


def main():
    """Run the Ultimatum Game simulation."""
    import os

    # Parse command-line arguments
    args = parse_arguments()

    # Update global configuration based on arguments
    global TOTAL_AMOUNT, PROPOSER_ONLY_MODE, SELF_DESCRIPTION, OPPONENT_DESCRIPTION, TRANSFER_RATE
    TOTAL_AMOUNT = args.total_amount
    PROPOSER_ONLY_MODE = args.proposer_only
    SELF_DESCRIPTION = args.self_description
    OPPONENT_DESCRIPTION = args.opponent_description
    TRANSFER_RATE = args.transfer_rate

    # Create test_results directory if it doesn't exist
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)

    # Generate parameter-based filename
    # Format player indices
    player_indices = args.player_indices
    if len(player_indices) > 2 and player_indices == list(range(player_indices[0], player_indices[-1] + 1)):
        # It's a continuous range
        players_str = f"p{player_indices[0]}-{player_indices[-1]}"
    elif len(player_indices) <= 3:
        # Short list, show all
        players_str = f"p{'_'.join(map(str, player_indices))}"
    else:
        # Long list, show count
        players_str = f"p{len(player_indices)}players"

    # Format games and transfer rate
    games_str = f"g{args.games}"
    rate_str = f"r{args.transfer_rate}"

    # Build base filename
    base_name = f"ultimatum_{players_str}_{games_str}_{rate_str}"

    # Find available filename (add number if exists)
    counter = 0
    while True:
        if counter == 0:
            csv_filename = f"{results_dir}/{base_name}.csv"
            json_filename = f"{results_dir}/{base_name}.json"
        else:
            csv_filename = f"{results_dir}/{base_name}_{counter}.csv"
            json_filename = f"{results_dir}/{base_name}_{counter}.json"

        # Check if either file exists
        if not os.path.exists(csv_filename) and not os.path.exists(json_filename):
            break
        counter += 1

    # Run simulation with parsed arguments
    simulation_results = run_simulation(
        player_indices=args.player_indices,
        games_per_direction=args.games,
        personas_file=args.personas_file,
        verbose=args.verbose
    )

    # Save results to CSV
    save_results_to_csv(
        simulation_results["results"], filename=csv_filename
    )

    # Optionally save full results to JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(simulation_results, f, indent=2)

    print(f"Results saved to {csv_filename}")
    print(f"Full results saved to {json_filename}")


if __name__ == "__main__":
    main()
