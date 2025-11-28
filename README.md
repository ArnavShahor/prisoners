# Ultimatum Game with LLM Agents

An experimental framework for studying economic decision-making by AI agents with diverse personalities in the Ultimatum Game.

## Overview

This project simulates one-shot Ultimatum Games between AI agents powered by large language models (LLMs). Each agent has a unique personality profile (demographics, background, traits) and makes decisions about how to split money with another agent.

**Key Features:**
- 100 pre-defined agent personalities with detailed backgrounds
- Support for multiple LLM providers (Anthropic, Apple internal)
- Token usage tracking and cost estimation
- Rich metadata capture (reasoning, token counts, failures)
- CSV and JSON output with personality distance metrics

## Quick Start

### Installation

1. Install Python 3.10+
2. Install required packages:
```bash
pip install anthropic  # For Anthropic API
```

### Running the Simulation

**Using Anthropic API:**
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python ultimatum_game.py
```

The simulation will run in test mode by default (6 agents, 90 games) and save results to `test_results/`.

## Project Structure

```
├── ultimatum_game.py      # Main Ultimatum Game implementation
├── prisoners_dilemma.py   # Prisoner's Dilemma game
├── llm_api.py            # LLM provider router
├── anthropic_api.py      # Anthropic API implementation
├── Personas.json         # 100 agent personality profiles
├── measure_cost.py       # Cost estimation utility
└── ULTIMATUM_GAME_PLAN.md # Detailed design documentation
```

## Configuration

Edit configuration constants in `ultimatum_game.py`:

```python
TEST_MODE = True              # Test mode (6 agents) or full mode (100 agents)
TEST_AGENT_INDICES = [24, 65, 40, 59, 70, 98]  # Which agents to use
TEST_GAMES_PER_DIRECTION = 3  # Games per agent pair
PROPOSER_ONLY_MODE = True     # Skip responder decision (faster)
```

### Environment Variables

- `LLM_PROVIDER` - API provider: `anthropic` or `apple` (default: `apple`)
- `ANTHROPIC_API_KEY` - Your Anthropic API key

## How It Works

1. **Load Personas**: 100 agents with unique personalities from `Personas.json`
2. **Agent Selection**: In test mode, uses a subset of 6 agents
3. **Game Loop**: Each agent plays as proposer against all others
4. **LLM Decision**: Agent decides how to split $100 (e.g., "I offer you $40")
5. **Results**: Saves detailed JSON and CSV files with offers, reasoning, and token usage

### Example Output

```
ultimatum_results_20241128_143522.json  # Full game data
ultimatum_results_20241128_143522.csv   # Simplified analysis
```

## The Ultimatum Game

In the Ultimatum Game:
- **Proposer** offers a split of $100 (e.g., "$60 for me, $40 for you")
- **Responder** accepts (both get money) or rejects (both get $0)
- Classic test of fairness, altruism, and strategic reasoning

## License

This project is for research and educational purposes.

## Related Files

- See `ULTIMATUM_GAME_PLAN.md` for detailed implementation notes
- See `measure_cost.py` for API cost estimation tools
