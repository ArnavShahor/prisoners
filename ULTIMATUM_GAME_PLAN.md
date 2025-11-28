# Plan: Convert Prisoner's Dilemma to Ultimatum Game

## Overview
Transform the existing `prisoners_dilemma.py` into a new `ultimatum_game.py` that plays one-shot ultimatum games between AI agents with different personalities.

## Game Rules - Ultimatum Game

### Structure
1. **Two players**: Proposer and Responder
2. **One-shot game**: Single interaction (no repeated rounds)
3. **Total amount**: 100 points to be divided

### Gameplay Flow
1. **Proposer** is given the Responder's personality profile
2. **Proposer** decides how to split 100 points (e.g., 60 for self, 40 for responder)
3. **Responder** sees the offer and the Proposer's personality
4. **Responder** chooses: ACCEPT or REJECT
   - **If ACCEPT**: Both players get their proposed amounts
   - **If REJECT**: Both players get 0

## Key Changes Required

### 1. Game Mechanics Replacement
**Remove:**
- 100-round iterated gameplay
- History tracking across rounds
- Cooperation/Defection (C/D) decision logic
- Payoff matrix for repeated interactions

**Add:**
- Single-shot ultimatum game structure
- Proposer makes offer (splits 100 points)
- Responder accepts or rejects
- Payoff calculation: accepted = proposed split, rejected = (0, 0)

### 2. Personality System Integration

**Data Source:** `Personas.json` (100 agents)

**Agent Attributes:**
- `player_number`: Unique ID
- `name`: Full name
- `gender`: Male/Female
- `age`: Integer
- `location`: Country
- `job`: Occupation
- `description`: Rich personality description
- `traits`: Dictionary with 5 traits
  - `FS`: Fair/Social orientation (0-1)
  - `GR`: Greed/Self-interest (0-1)
  - `RA`: Risk Aversion (0-1)
  - `SS`: Social Sensitivity (0-1)
  - `ST`: Strategic Thinking (0-1)

**Integration:**
- Load all 100 personas from JSON
- Pass personality context to AI agents (excluding traits)
- Include name, age, job, and description in prompts
- **EXCLUDE traits from prompts** - traits are only for post-game analysis

### 3. Agent Class Redesign

**Current:** `PrisonersDilemmaAgent`
- Makes repeated C/D decisions
- Tracks history
- Single strategy throughout

**New:** `UltimatumAgent`
- Initialized with a personality from `Personas.json`
- Can play BOTH roles (proposer and responder)

**Methods:**
```python
class UltimatumAgent:
    def __init__(self, persona: dict, model: str):
        # Initialize with persona data

    def make_proposal(self, responder_persona: dict) -> tuple[int, dict]:
        """
        Proposer role: Decide how to split 100 points

        Args:
            responder_persona: Full personality dict of the responder

        Returns:
            (offer_to_responder, usage_info)
            offer_to_responder: int between 0-100 (responder gets this, proposer gets 100-this)
        """

    def respond_to_offer(self, proposer_persona: dict, offer: int) -> tuple[str, dict]:
        """
        Responder role: Accept or reject the offer

        Args:
            proposer_persona: Full personality dict of the proposer
            offer: Amount offered to this agent (0-100)

        Returns:
            ("accept" or "reject", usage_info)
        """
```

### 4. Prompt Engineering

#### Proposer Prompt Structure
```
You are {YOUR_NAME}, age {YOUR_AGE}, works as {YOUR_JOB}.
Your description: {YOUR_DESCRIPTION}

You are playing the Ultimatum Game with {RESPONDER_NAME}.
About {RESPONDER_NAME}:
- Age: {RESPONDER_AGE}
- Job: {RESPONDER_JOB}
- Description: {RESPONDER_DESCRIPTION}

RULES:
You must split 100 points between yourself and {RESPONDER_NAME}.
You propose how much {RESPONDER_NAME} gets (0-100).
You will get (100 - offer).

If {RESPONDER_NAME} accepts: both get the proposed amounts.
If {RESPONDER_NAME} rejects: both get 0.

Decide how much to offer {RESPONDER_NAME}.

Respond ONLY with JSON:
{
  "reasoning": "your strategic thinking",
  "offer": <number between 0 and 100>
}
```

#### Responder Prompt Structure
```
You are {YOUR_NAME}, age {YOUR_AGE}, works as {YOUR_JOB}.
Your description: {YOUR_DESCRIPTION}

You are playing the Ultimatum Game with {PROPOSER_NAME}.
About {PROPOSER_NAME}:
- Age: {PROPOSER_AGE}
- Job: {PROPOSER_JOB}
- Description: {PROPOSER_DESCRIPTION}

SITUATION:
{PROPOSER_NAME} has proposed to split 100 points:
- You get: {OFFER}
- {PROPOSER_NAME} gets: {100 - OFFER}

RULES:
If you ACCEPT: you get {OFFER}, {PROPOSER_NAME} gets {100 - OFFER}
If you REJECT: both of you get 0

Decide whether to accept or reject.

Respond ONLY with JSON:
{
  "reasoning": "your thinking process",
  "decision": "accept" or "reject"
}
```

### 5. Game Simulation Structure

#### Pairing Strategy
**Option A:** All possible pairs (recommended for research)
- 100 agents → 4,950 unique pairs
- Each pair plays TWICE (A→B, then B→A)
- Total: 9,900 games

**Option B:** Random subset
- Sample N pairs randomly
- Useful for testing or cost control

#### Game Flow
```python
for pair (agent_i, agent_j) in all_pairs:
    # Game 1: i proposes, j responds
    game1 = play_game(proposer=agent_i, responder=agent_j)

    # Game 2: j proposes, i responds
    game2 = play_game(proposer=agent_j, responder=agent_i)
```

#### Single Game Flow
```python
def play_game(proposer, responder):
    # 1. Proposer makes offer
    offer = proposer.make_proposal(responder.persona)

    # 2. Responder decides
    decision = responder.respond_to_offer(proposer.persona, offer)

    # 3. Calculate payoffs
    if decision == "accept":
        proposer_payoff = 100 - offer
        responder_payoff = offer
    else:
        proposer_payoff = 0
        responder_payoff = 0

    return game_results
```

### 6. Results & Analysis

#### Data to Track (per game)
- `game_id`: Unique identifier
- `proposer_name`: Name of proposer
- `responder_name`: Name of responder
- `proposer_idx`: Index in persona array
- `responder_idx`: Index in persona array
- `offer`: Amount offered to responder (0-100)
- `decision`: "accept" or "reject"
- `proposer_payoff`: Final payoff for proposer
- `responder_payoff`: Final payoff for responder
- `proposer_reasoning`: AI's reasoning for proposal
- `responder_reasoning`: AI's reasoning for decision
- `proposer_traits`: Dict of 5 traits
- `responder_traits`: Dict of 5 traits
- `proposer_age`, `proposer_gender`, `proposer_job`
- `responder_age`, `responder_gender`, `responder_job`
- `similarity`: Personality similarity score (optional)
- `proposer_tokens`: Token usage for proposal
- `responder_tokens`: Token usage for response
- `total_tokens`: Combined token usage

#### Output Format
**Primary:** CSV file (`ultimatum_results.csv`)
**Optional:** JSON file with full details

#### Analysis Capabilities
- Acceptance rates by offer amount
- Correlation between personality traits and offers
- Correlation between personality traits and acceptance
- Gender/age/job effects
- Personality similarity effects
- Token usage and cost analysis

### 7. Files to Create/Modify

#### New Files
- **`ultimatum_game.py`**: Main game implementation
  - Based on structure of `prisoners_dilemma.py`
  - Replace game logic
  - Keep token tracking system
  - Keep verbose output and timing

#### Existing Files (No Changes)
- **`token_counter.py`**: LLM query library (reuse as-is)
- **`Personas.json`**: Personality data (100 agents)

#### Reference Files
- **`Ultimatum_script.py`**: Template for simulation loop and CSV export
- **`prisoners_dilemma.py`**: Template for structure, token tracking, timing

## Implementation Steps

### Phase 1: Core Classes
1. Create `UltimatumAgent` class
   - Load persona from dict
   - Implement `make_proposal()` method
   - Implement `respond_to_offer()` method
   - Track token usage (reuse existing pattern)

2. Create `UltimatumGame` class
   - Single game between two agents
   - Role assignment (proposer/responder)
   - Payoff calculation
   - Results dictionary

### Phase 2: Simulation Engine
3. Load personas from `Personas.json`
4. Generate all pair combinations
5. Implement game loop (each pair plays twice)
6. Collect results with full metadata

### Phase 3: Output & Analysis
7. Export to CSV format
8. Add summary statistics
9. Calculate personality similarity (optional)
10. Token usage and cost reporting

### Phase 4: Testing & Validation
11. Test with 2-3 agents first
12. Validate JSON parsing from LLM
13. Check payoff calculations
14. Run full simulation

## Configuration Parameters

```python
# Simulation settings
PERSONAS_JSON_PATH = "Personas.json"
OUTPUT_CSV = "ultimatum_results.csv"
OUTPUT_JSON = "ultimatum_results.json"  # optional

# Game settings
TOTAL_AMOUNT = 100  # points to split

# LLM settings
MODEL = "openai/aws:anthropic.claude-sonnet-4-20250514-v1:0"
MAX_TOKENS = 512

# Simulation scope - INITIAL TEST
RUN_MODE = "test"  # "test" for 2 agents, "subset" for random sample, "full" for all pairs
TEST_AGENT_INDICES = [5, 21]  # Example: Aisha Yusuf (nurse) vs Victor Petrov (economist)
RANDOM_SAMPLE_SIZE = 100  # if RUN_MODE is "subset"
RANDOM_SEED = 42
```

## Expected Output Example

### Console Output
```
======================================================================
ULTIMATUM GAME SIMULATION - TEST MODE
======================================================================
Loaded 100 personas from Personas.json
Running TEST MODE: 2 agents → 2 games
======================================================================

Selected agents:
  1. Aisha Yusuf - 28-year-old nurse in Kenya
     "Empathetic and people-oriented, spends long hours caring for patients..."

  2. Victor Petrov - 45-year-old economist from Russia
     "Focuses on metrics and profitability, models trade-offs carefully..."

======================================================================

Game 1/2: Aisha Yusuf (proposer) → Victor Petrov (responder)
  Aisha's reasoning: "As a nurse who values fairness and caring for others..."
  Offer: 48
  Victor's reasoning: "From an economic perspective, 48 is reasonable..."
  Decision: accept
  Payoffs: Aisha=52, Victor=48

Game 2/2: Victor Petrov (proposer) → Aisha Yusuf (responder)
  Victor's reasoning: "Optimal strategy suggests offering minimum acceptable..."
  Offer: 35
  Aisha's reasoning: "While this feels somewhat unfair, it's still..."
  Decision: accept
  Payoffs: Victor=65, Aisha=35

======================================================================
TEST SIMULATION COMPLETE
======================================================================
Total games: 2
Total accepted: 2 (100%)
Total rejected: 0 (0%)
Average offer: 41.5
Average proposer payoff: 58.5
Average responder payoff: 41.5

Token Usage: 2,847 tokens
Estimated Cost: $0.02

Results saved to: ultimatum_results.csv
======================================================================
```

### CSV Output Structure
```csv
game_id,proposer_name,responder_name,offer,decision,proposer_payoff,responder_payoff,proposer_FS,proposer_GR,responder_FS,responder_GR,...
1,Daniel Cohen,Maria Santos,45,accept,55,45,0.63,0.41,0.74,0.29,...
2,Maria Santos,Daniel Cohen,50,accept,50,50,0.74,0.29,0.63,0.41,...
...
```

## Questions to Resolve

1. **Personality similarity calculation**: Use cosine similarity (as in `Ultimatum_script.py`) or skip?
2. **Run scope**: ✅ **START WITH 2 EXAMPLE PLAYERS** - Full run later after validation
3. **Output format preference**: CSV only, or also JSON with full reasoning text?
4. **Verbose mode**: Print every game or just summary statistics?
5. **Cost considerations**: Start small, scale up after initial results look good

## Initial Test Run Configuration

**Starting approach:**
- Pick 2 interesting/contrasting personas from the 100 available
- Run 2 games: A→B and B→A
- Validate that:
  - LLM prompts work correctly
  - JSON parsing works
  - Offers are reasonable (0-100)
  - Accept/reject logic works
  - Results are saved correctly
- Print full reasoning for both agents
- Review results before scaling up

**Example pair suggestions:**
- **High fairness vs High greed**: e.g., Aisha Yusuf (nurse, FS=0.85) vs Victor Petrov (economist, GR=0.81)
- **Risk-averse vs Risk-taking**: e.g., Thomas Berg (ship captain, RA=0.86) vs Liam Johnson (student, RA=0.31)
- **Strategic vs Impulsive**: e.g., Noah Williams (data analyst, ST=0.88) vs Sofia Marin (dancer, ST=0.39)

## Notes
- Reuse token tracking from Prisoner's Dilemma (well-implemented)
- Keep timing/performance measurement patterns
- Model remains: `openai/aws:anthropic.claude-sonnet-4-20250514-v1:0`
- Can switch to Anthropic direct API later (per earlier discussion)
