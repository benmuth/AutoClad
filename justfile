# STS-AI Justfile - Neural Network Training and Development
# Usage: just <command>

# Default recipe that displays available commands
default:
    @just --list

# Build configuration
BUILD_DIR := "build"

# ========================================
# NEURAL NETWORK TRAINING PIPELINE
# ========================================
# Complete workflow: randomize-scenarios → generate-data → parse-data → train → run-neural-agent

# Step 1: Generate randomized battle scenarios for training data
# Example: just randomize-scenarios 20
randomize-scenarios count="20":
    @echo "Generating {{count}} randomized scenarios per base scenario..."
    uv run randomize_scenarios.py --count {{count}}

# Step 2: Generate training data by running battles and saving snapshots
# Example: just generate-data
generate-data:
    @echo "Running simple agent battles to generate training data..."
    just run battle-agent simple --snapshot --scenario=jaw_worm

# Step 3: Parse battle snapshots into numpy arrays for training
# Example: just parse-data
parse-data:
    @echo "Parsing battle snapshots into training data..."
    cd AutoClad && uv run data_parser.py

# Step 4: Train the neural network model with custom arguments
# Example: just train --plot --early-stopping --epochs 300
train *ARGS:
    @echo "Training neural network model..."
    cd AutoClad && uv run main.py {{ARGS}}

# Step 5: Run the trained neural agent via CommunicationMod protocol
# Example: just run-neural-agent
run-neural-agent:
    @echo "Running trained neural agent..."
    cd AutoClad && uv run neural_agent.py

# Complete pipeline from scratch (all steps combined)
# Example: just pipeline 20
pipeline count="20":
    @echo "=== Starting complete ML pipeline ==="
    just randomize-scenarios {{count}}
    just generate-data
    just parse-data
    just train --plot --early-stopping
    @echo "=== Pipeline complete! Test with: just run-neural-agent ==="

# ========================================
# BUILD COMMANDS
# ========================================

# Build C++ executables with options
# target: all (default), battle-agent, battle, or any executable name
# type: release (default), debug
# libtorch: false (default), true (requires LIBTORCH_PATH env var)
# Examples:
#   just build                           # Build all executables
#   just build battle-agent              # Build only battle-agent
#   just build all debug                 # Build all with debug symbols
#   just build battle-agent release true # Build battle-agent with LibTorch
build target="all" type="release" libtorch="false":
    #!/usr/bin/env bash
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}}

    # Set CMake flags based on build type
    if [ "{{type}}" = "debug" ]; then
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=\"-g -O0\""
    else
        CMAKE_FLAGS=""
    fi

    # Add LibTorch support if requested
    if [ "{{libtorch}}" = "true" ]; then
        if [ -z "${LIBTORCH_PATH}" ]; then
            echo "Error: LIBTORCH_PATH environment variable not set"
            exit 1
        fi
        CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH}"
    fi

    # Build
    eval cmake $CMAKE_FLAGS .. && make {{target}}

# Clean all build artifacts
clean:
    rm -rf {{BUILD_DIR}}
    find . -name "*.o" -delete
    find . -name "*.a" -delete

# ========================================
# RUNNING AGENTS AND TESTS
# ========================================

# Run C++ executables with options
# exe: main (interactive), battle, battle-agent, test, small-test
# For battle-agent, specify agent: simple, autoclad, neural
# Examples:
#   just run main                          # Interactive simulator
#   just run battle                        # Standalone battle
#   just run battle-agent simple           # Simple agent
#   just run battle-agent simple --snapshot --scenario=jaw_worm
#   just run battle-agent neural --scenario=jaw_worm
#   just run test agent_mt 1 1 0 1984 1 1 # Test with args
run exe *ARGS:
    #!/usr/bin/env bash
    if [ "{{exe}}" = "battle-agent" ]; then
        # Extract agent type from first arg
        agent="${1:-simple}"
        shift || true  # Remove agent from args

        if [ "$agent" = "neural" ]; then
            echo "Building battle-agent with LibTorch..."
            just build battle-agent release true
            DYLD_LIBRARY_PATH=${LIBTORCH_PATH}/lib ./{{BUILD_DIR}}/battle-agent --agent=$agent "$@"
        else
            just build battle-agent
            ./{{BUILD_DIR}}/battle-agent --agent=$agent "$@"
        fi
    else
        just build {{exe}}
        ./{{BUILD_DIR}}/{{exe}} {{ARGS}}
    fi

# ========================================
# GAME VERIFICATION (DEVELOPMENT)
# ========================================

# Replay game from save file with action sequence
# Used for verifying game logic correctness
# Example: just replay-save saves/ironclad_run.json actions.txt
replay-save savefile actionfile:
    just run test save {{savefile}} {{actionfile}}

# Replay game from seed and ascension with action sequence
# Used for verifying deterministic game behavior
# Example: just replay-seed 12345 0 actions.txt
replay-seed seed ascension actionfile:
    just run test replay {{seed}} {{ascension}} {{actionfile}}

# ========================================
# SCENARIO GENERATION (ONE-TIME SETUP)
# ========================================

# Generate base scenarios from replay data (ONE-TIME SETUP)
# This converts game replay JSON files into battle scenario files
# force: false (default), true (delete existing output directory)
# Example: just generate-scenarios data/2019-runs-json-ironclad/ battle/generated_scenarios/
# Example with force: just generate-scenarios data/2019-runs-json-ironclad/ battle/generated_scenarios/ true
generate-scenarios input_dir output_dir="battle/generated_scenarios/" force="false":
    #!/usr/bin/env bash
    if [ "{{force}}" = "true" ]; then
        uv run data/scenarios-from-runs.py {{input_dir}} {{output_dir}} --force
    else
        uv run data/scenarios-from-runs.py {{input_dir}} {{output_dir}}
    fi
