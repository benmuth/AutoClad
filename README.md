# AutoClad

Training neural network agents to play Slay the Spire using supervised learning.

This project uses [sts-lightspeed](https://github.com/gamerpuppy/sts_lightspeed), a high-performance RNG-accurate C++ simulator, to generate training data and evaluate agents. The ultimate goal is to build an agent that can eventually play Ascension 20 Ironclad.

## What This Project Does

This currently is a **supervised learning pipeline** that trains neural network agents to play Slay the Spire:

1. **Generate diverse battle scenarios** from real game data
2. **Run a baseline agent** to collect demonstrations
3. **Extract state-action pairs** from battle snapshots
4. **Train a neural network** to imitate the baseline agent's behavior
5. **Export the model** for C++ inference and evaluation

Currently focused on benchmarking different approaches.

## Quick Start

### Prerequisites

- C++17 compiler and CMake 3.10+ (for the sts-lightspeed simulator)
- [just](https://github.com/casey/just) (recommended for running commands)
- Python 3.8+ with [uv](https://github.com/astral-sh/uv)

### Build the Simulator

```bash
just build
```

## Training Pipeline

### 1. Generate Training Scenarios

Create randomized battle scenarios from real game data:

```bash
uv run randomize_scenarios.py --count 20
```

This creates variations of base scenarios by randomizing HP, deck composition, and RNG seeds.

### 2. Collect Demonstration Data

Run SimpleAgent on scenarios and capture battle snapshots:

```bash
just run-agent simple --snapshot --scenario=jaw_worm
```

Battle progression data is saved to `data/agent_battles/simpleagent/`.

### 3. Parse Training Data

Extract state-action pairs from battle snapshots:

```bash
cd AutoClad
uv run data_parser.py
```

Creates `jaw_worm_data.npz` with feature vectors (game state) and action labels (which card was played).

### 4. Train the Neural Network

```bash
cd AutoClad

# Train with plotting and early stopping
uv run main.py --plot --early-stopping

# Or with default settings
uv run main.py
```

The trained model is exported as `jaw_worm_model_traced.pt` (TorchScript format) for C++ inference.

### 5. Evaluate the Trained Agent

```bash
# Requires LibTorch
LIBTORCH_PATH=~/Downloads/libtorch just run-agent neural --scenario=jaw_worm
```

## Current Status

**Working:**
- Data generation from baseline agent demonstrations
- Neural network training on Jaw Worm encounters
- C++ inference with trained models

**In Progress:**
- Expanding to more encounter types
- Improving agent performance metrics
- Recording intermediate combat states for richer training data

## The Big Picture

The end goal is to train an agent that can play the full game at a high level. The approach:

1. **Start with battles**: If you can predict whether you'll win specific fights, you can make better decisions about everything else (pathing, card choices, shops, etc.)

2. **Generate quality data**: Available human gameplay data is incomplete and outdated. Instead, use tree search and simpler agents to generate training data.

3. **Scale up with RL**: Once we have a decent baseline from supervised learning, use reinforcement learning to reach superhuman play.

## Project Structure

```
├── AutoClad/                    # Neural network training code (Python)
│   ├── main.py                  # Training script
│   ├── data_parser.py           # Parse battle snapshots to training data
│   └── *.pth, *.pt              # Trained models
├── randomize_scenarios.py       # Generate randomized battle scenarios
├── battle/
│   ├── generated_scenarios/     # Base scenarios from real games
│   └── randomized_scenarios/    # Training scenario variations
├── data/agent_battles/          # Battle snapshots from agent runs
└── [sts-lightspeed files]       # C++ simulator (apps/, src/, include/, etc.)

