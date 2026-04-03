# Ultimatum Game with LLM Agents

An experimental framework for studying economic decision-making by AI agents with diverse personalities in the Ultimatum Game.

## Overview

This project simulates one-shot Ultimatum Games between AI agents powered by large language models (LLMs). Each agent has a unique personality profile (demographics, background, traits) and makes decisions about how to split money with another agent.

**Key Features:**
- 100 pre-defined agent personalities with detailed backgrounds
- Anthropic Claude API integration
- Token usage tracking and cost estimation
- Rich metadata capture (reasoning, token counts, failures)
- CSV and JSON output with personality distance metrics

## Quick Start

### Installation

1. Install Python 3.10+
2. Install required packages:
```bash
pip install anthropic
```

### Running the Simulation

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.ultimatum_game
```

The simulation will run in test mode by default (6 agents, 90 games) and save results to `results/`.

**Parallel execution:**
```bash
python -m src.ultimatum_game_parallel --players 0-5 --games 5 --workers 10
```

## Project Structure

```
├── src/                              # Core simulation code
│   ├── anthropic_api.py              # Anthropic API wrapper
│   ├── ultimatum_game.py             # Main Ultimatum Game implementation
│   ├── ultimatum_game_parallel.py    # Parallel execution version
│   └── prisoners_dilemma.py          # Prisoner's Dilemma game
│
├── analysis/                         # Analysis & utility scripts
│   ├── analyze_cluster_generosity.py
│   ├── analyze_cluster_linear_models.py
│   ├── analyze_job_similarity_offers.py
│   ├── combine_runs.py
│   ├── measure_cost.py
│   ├── embedded_jobs.py
│   └── test_rate_limiting.py
│
├── data/                             # Input JSON data
│   ├── Personas.json
│   ├── Personas_Jobs.json
│   └── *.json (embeddings, similarities, clusters)
│
├── results/                          # Simulation outputs (gitignored)
├── visualizations/                   # Generated plots
└── docs/                             # Paper and documentation
```

## Configuration

Configuration is controlled via command-line arguments:

```bash
python -m src.ultimatum_game --players 0-5 --games 5 --proposer-only
python -m src.ultimatum_game --personas-file data/Personas_Jobs.json --transfer-rate 0.8
```

### Environment Variables

- `ANTHROPIC_API_KEY` - Your Anthropic API key

## How It Works

1. **Load Personas**: 100 agents with unique personalities from `data/Personas_Jobs.json`
2. **Agent Selection**: Select a subset of agents by index
3. **Game Loop**: Each agent plays as proposer against all others
4. **LLM Decision**: Agent decides how to split $100 (e.g., "I offer you $40")
5. **Results**: Saves detailed JSON and CSV files with offers, reasoning, and token usage

## The Ultimatum Game

In the Ultimatum Game:
- **Proposer** offers a split of $100 (e.g., "$60 for me, $40 for you")
- **Responder** accepts (both get money) or rejects (both get $0)
- Classic test of fairness, altruism, and strategic reasoning

## License

This project is for research and educational purposes.
