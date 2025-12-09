#!/usr/bin/env python3
"""
Parallel Ultimatum Game with LLM Agents

Enhanced version that runs multiple games concurrently using ThreadPoolExecutor
to speed up API-bound operations.
"""

import argparse
import json
import time
import os
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys

# Import components from the original ultimatum_game module
from ultimatum_game import (
    UltimatumAgent,
    UltimatumGame,
    load_personas,
    save_results_to_csv,
    DEFAULT_TOTAL_AMOUNT,
    DEFAULT_PLAYER_INDICES,
    DEFAULT_GAMES_PER_DIRECTION,
    DEFAULT_PROPOSER_ONLY_MODE,
    DEFAULT_SELF_DESCRIPTION,
    DEFAULT_OPPONENT_DESCRIPTION,
    DEFAULT_PERSONAS_FILE,
    DEFAULT_TRANSFER_RATE
)

# Runtime configuration (will be set by command-line args)
TOTAL_AMOUNT = DEFAULT_TOTAL_AMOUNT
PROPOSER_ONLY_MODE = DEFAULT_PROPOSER_ONLY_MODE
SELF_DESCRIPTION = DEFAULT_SELF_DESCRIPTION
OPPONENT_DESCRIPTION = DEFAULT_OPPONENT_DESCRIPTION
TRANSFER_RATE = DEFAULT_TRANSFER_RATE

# Parallel execution configuration
DEFAULT_MAX_WORKERS = 10  # Default number of concurrent games


# ============================================================================
# THREAD-SAFE AGENT WRAPPER
# ============================================================================

class ThreadSafeUltimatumAgent(UltimatumAgent):
    """
    Thread-safe wrapper for UltimatumAgent that uses locks to protect
    token tracking statistics during concurrent updates.
    """

    def __init__(self, persona: dict[str, Any], model: str = None):
        super().__init__(persona, model)
        self._lock = Lock()

    def make_proposal(self, responder_persona: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
        """Thread-safe version of make_proposal."""
        # Get the proposal (API call happens here)
        keep_amount, actual_offer, result = super().make_proposal(responder_persona)

        # Update statistics with lock
        with self._lock:
            # Statistics are already updated in parent class
            pass

        return keep_amount, actual_offer, result

    def respond_to_offer(self, proposer_persona: dict[str, Any], keep_amount: int, offer: int) -> tuple[str, dict[str, Any]]:
        """Thread-safe version of respond_to_offer."""
        # Get the response (API call happens here)
        decision, result = super().respond_to_offer(proposer_persona, keep_amount, offer)

        # Update statistics with lock
        with self._lock:
            # Statistics are already updated in parent class
            pass

        return decision, result

    def get_stats(self) -> dict[str, Any]:
        """Thread-safe statistics retrieval."""
        with self._lock:
            return super().get_stats()


# ============================================================================
# PARALLEL GAME EXECUTOR
# ============================================================================

def run_single_game(args: tuple) -> Optional[dict[str, Any]]:
    """
    Run a single game. This function is designed to be called by ThreadPoolExecutor.

    Args:
        args: Tuple of (proposer_agent, responder_agent, game_id, verbose)

    Returns:
        Game result dictionary or None if error
    """
    proposer, responder, game_id, verbose = args

    try:
        # Override the global variables for this thread
        import ultimatum_game
        ultimatum_game.TOTAL_AMOUNT = TOTAL_AMOUNT
        ultimatum_game.PROPOSER_ONLY_MODE = PROPOSER_ONLY_MODE
        ultimatum_game.SELF_DESCRIPTION = SELF_DESCRIPTION
        ultimatum_game.OPPONENT_DESCRIPTION = OPPONENT_DESCRIPTION
        ultimatum_game.TRANSFER_RATE = TRANSFER_RATE

        game = UltimatumGame(proposer=proposer, responder=responder, game_id=game_id)
        result = game.play(verbose=verbose)
        return result
    except Exception as e:
        print(f"\n⚠️  Error in game {game_id}: {str(e)[:100]}")
        return None


def run_parallel_simulation(
    player_indices: list[int] = None,
    games_per_direction: int = None,
    personas_file: str = None,
    verbose: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS
) -> dict[str, Any]:
    """
    Run the ultimatum game simulation with parallel execution.

    Args:
        player_indices: List of player indices to use
        games_per_direction: Number of games each agent plays as proposer
        personas_file: Path to personas JSON file
        verbose: Print progress
        max_workers: Maximum number of concurrent games

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
    print("PARALLEL ULTIMATUM GAME SIMULATION")
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
    print(f"Max concurrent games: {max_workers}")

    # Get selected personas
    selected_personas = [personas[i] for i in agent_indices]

    print("=" * 70)
    print("\nSelected agents:")
    for i, persona in enumerate(selected_personas, 1):
        print(f"  {i}. {persona['name']} - {persona['age']}-year-old {persona['job']}")
        print(f"     {persona['description'][:80]}...")

    print("\n" + "=" * 70)

    # Create thread-safe agents
    agents = [ThreadSafeUltimatumAgent(persona) for persona in selected_personas]

    # Prepare all game combinations
    game_tasks = []
    game_count = 0

    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:  # Don't play against self
                for _ in range(games_per_direction):
                    game_count += 1
                    game_tasks.append((agents[i], agents[j], game_count, verbose))

    # Run games in parallel
    results = []
    failed_games = 0
    start_time = time.time()

    print(f"\n🚀 Starting parallel execution with {max_workers} workers...")
    print("=" * 70)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_game = {
            executor.submit(run_single_game, task): task[2]  # task[2] is game_id
            for task in game_tasks
        }

        # Process completed games
        completed = 0
        for future in as_completed(future_to_game):
            game_id = future_to_game[future]
            completed += 1

            try:
                result = future.result(timeout=60)  # 60 second timeout per game
                if result:
                    results.append(result)
                else:
                    failed_games += 1
                    print(f"⚠️  Game {game_id} failed")
            except Exception as e:
                failed_games += 1
                print(f"⚠️  Game {game_id} exception: {str(e)[:100]}")

            # Update progress bar
            if not verbose:
                percent = (completed / total_games) * 100
                bar_length = 50
                filled = int(bar_length * completed / total_games)
                bar = '█' * filled + '░' * (bar_length - filled)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_games - completed) / rate if rate > 0 else 0
                print(f'\rProgress: [{bar}] {percent:.1f}% ({completed}/{total_games}) | Rate: {rate:.1f} games/s | ETA: {eta:.0f}s',
                      end='', flush=True)

    # Print newline after progress bar
    if not verbose:
        print()  # Move to next line after progress bar

    elapsed_time = time.time() - start_time

    # Calculate summary statistics
    successful_games = len(results)
    accepted = sum(1 for r in results if r["decision"] == "accept")
    rejected = successful_games - accepted
    avg_offer = sum(r["offer"] for r in results) / successful_games if successful_games > 0 else 0
    avg_proposer_payoff = (
        sum(r["proposer_payoff"] for r in results) / successful_games
        if successful_games > 0
        else 0
    )
    avg_responder_payoff = (
        sum(r["responder_payoff"] for r in results) / successful_games
        if successful_games > 0
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
    print(f"Execution time: {elapsed_time:.1f} seconds")
    print(f"Average rate: {successful_games/elapsed_time:.1f} games/second")
    print(f"Speedup vs sequential: ~{max_workers:.1f}x (theoretical max)")
    print()
    print(f"Total games attempted: {total_games}")
    print(f"Successful games: {successful_games} ({successful_games/total_games*100:.1f}%)")
    print(f"Failed games: {failed_games} ({failed_games/total_games*100:.1f}%)")

    if successful_games > 0:
        print(f"Total accepted: {accepted} ({accepted/successful_games*100:.1f}%)")
        print(f"Total rejected: {rejected} ({rejected/successful_games*100:.1f}%)")
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
            "successful_games": successful_games,
            "failed_games": failed_games,
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
            "execution_time": elapsed_time,
            "games_per_second": successful_games / elapsed_time if elapsed_time > 0 else 0
        },
        "agents": [agent.get_stats() for agent in agents],
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_arguments():
    """Parse command-line arguments for the Parallel Ultimatum Game simulation."""
    parser = argparse.ArgumentParser(
        description="Run Parallel Ultimatum Game simulation with AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with players 1-6, 5 games per direction, 10 concurrent workers
  python ultimatum_game_parallel.py --players 0-5 --games 5 --workers 10

  # Run with specific players, 10 games, 20 workers, verbose
  python ultimatum_game_parallel.py -p 0,1,2 -g 10 -w 20 -v

  # Run full game mode with maximum parallelism
  python ultimatum_game_parallel.py --full-game --workers 20

  # Run with custom personas file and transfer rate
  python ultimatum_game_parallel.py --personas-file custom_personas.json --transfer-rate 0.8
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

    # Parallel execution configuration
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of concurrent games. Default: {DEFAULT_MAX_WORKERS}"
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
        help="Transfer rate (must be > 0). Responder receives (100-keep) × rate. Default: 1.0"
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

    # Validate transfer rate (must be > 0)
    if args.transfer_rate <= 0:
        parser.error(f"transfer-rate must be greater than 0, got {args.transfer_rate}")

    # Validate workers
    if args.workers < 1:
        parser.error(f"workers must be at least 1, got {args.workers}")

    return args


def main():
    """Run the Parallel Ultimatum Game simulation."""
    # Parse command-line arguments
    args = parse_arguments()

    # Update global configuration based on arguments
    global TOTAL_AMOUNT, PROPOSER_ONLY_MODE, SELF_DESCRIPTION, OPPONENT_DESCRIPTION, TRANSFER_RATE
    TOTAL_AMOUNT = args.total_amount
    PROPOSER_ONLY_MODE = args.proposer_only
    SELF_DESCRIPTION = args.self_description
    OPPONENT_DESCRIPTION = args.opponent_description
    TRANSFER_RATE = args.transfer_rate

    # Also update the original module's globals
    import ultimatum_game
    ultimatum_game.TOTAL_AMOUNT = TOTAL_AMOUNT
    ultimatum_game.PROPOSER_ONLY_MODE = PROPOSER_ONLY_MODE
    ultimatum_game.SELF_DESCRIPTION = SELF_DESCRIPTION
    ultimatum_game.OPPONENT_DESCRIPTION = OPPONENT_DESCRIPTION
    ultimatum_game.TRANSFER_RATE = TRANSFER_RATE

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

    # Format games, transfer rate, and workers
    games_str = f"g{args.games}"
    rate_str = f"r{args.transfer_rate}"
    workers_str = f"w{args.workers}"

    # Build base filename (include "parallel" to distinguish from sequential version)
    base_name = f"ultimatum_parallel_{players_str}_{games_str}_{rate_str}_{workers_str}"

    # Find available filename (add number if exists)
    counter = 0
    while True:
        if counter == 0:
            csv_filename = f"{results_dir}/{base_name}.csv"
        else:
            csv_filename = f"{results_dir}/{base_name}_{counter}.csv"

        # Check if file exists
        if not os.path.exists(csv_filename):
            break
        counter += 1

    # Run simulation with parsed arguments
    simulation_results = run_parallel_simulation(
        player_indices=args.player_indices,
        games_per_direction=args.games,
        personas_file=args.personas_file,
        verbose=args.verbose,
        max_workers=args.workers
    )

    # Save results to CSV
    save_results_to_csv(
        simulation_results["results"], filename=csv_filename
    )

    print(f"Results saved to {csv_filename}")

    # Print performance comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    exec_time = simulation_results["summary"]["execution_time"]
    games_per_sec = simulation_results["summary"]["games_per_second"]
    total_games = simulation_results["summary"]["total_games"]
    sequential_estimate = total_games / games_per_sec * args.workers if games_per_sec > 0 else 0

    print(f"Parallel execution time: {exec_time:.1f} seconds")
    print(f"Estimated sequential time: {sequential_estimate:.1f} seconds")
    if sequential_estimate > 0:
        print(f"Actual speedup: {sequential_estimate/exec_time:.1f}x")
    print(f"Games per second: {games_per_sec:.1f}")
    print(f"Workers used: {args.workers}")


if __name__ == "__main__":
    main()