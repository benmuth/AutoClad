# STS-AI Justfile - Common project commands
# Usage: just <command>

# Default recipe that displays available commands
default:
    @just --list

# Build configuration
BUILD_DIR := "build"
TEST_BUILD_DIR := "tests/build"

# === Build Commands ===

# Build the main project (creates main, test, small-test, and battle executables)
build:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake .. && make

# Clean and rebuild everything
rebuild: clean build

# Clean build artifacts
clean:
    rm -r {{BUILD_DIR}}
    rm -r {{TEST_BUILD_DIR}}

# Build optimized release version
build-release:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake -DCMAKE_BUILD_TYPE=Release .. && make

# Build debug version with debug symbols
build-debug:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g -O0" .. && make

# === Test Commands ===

# Build test infrastructure (snapshot generator)
build-tests:
    mkdir -p {{TEST_BUILD_DIR}}
    #!/usr/bin/env bash
    cd {{TEST_BUILD_DIR}} && cmake .. && make

# Run the main test suite (requires arguments)
test *ARGS:
    just build
    ./{{BUILD_DIR}}/test {{ARGS}}

# Run small test executable
test-small:
    just build
    ./{{BUILD_DIR}}/small-test

# Run multi-threaded agent test (threads, depth, ascension, seed, count, print)
test-agent threads depth ascension seed count print="1":
    just build
    ./{{BUILD_DIR}}/test agent_mt {{threads}} {{depth}} {{ascension}} {{seed}} {{count}} {{print}}

# Run simple agent test (threads, seed, count, print)
test-simple-agent threads seed count print="1":
    just build
    ./{{BUILD_DIR}}/test simple_agent_mt {{threads}} {{seed}} {{count}} {{print}}

# Run neural network agent test (threads, seed, count, print) - requires LibTorch
test-neural-agent threads seed count print="1":
    just build-battle-agent
    ./{{BUILD_DIR}}/test neural_agent_mt {{threads}} {{seed}} {{count}} {{print}}

# Build with LibTorch support (set LIBTORCH_PATH environment variable)
build-with-libtorch:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_PATH}" .. && make battle-agent

# Train neural network model and export for C++
train-neural-model:
    cd AutoClad && uv run main.py

# Run MCTS test from save file
test-mcts savefile simulations:
    just build
    ./{{BUILD_DIR}}/test mcts_save {{savefile}} {{simulations}}

# Generate combat snapshot for testing
snapshot scenario output:
    just build-tests
    ./{{TEST_BUILD_DIR}}/snapshot_generator {{scenario}} {{output}}
    
# Generate basic combat snapshot
snapshot-basic:
    just build-tests
    mkdir -p tests/snapshots/basic_combat
    ./{{TEST_BUILD_DIR}}/snapshot_generator tests/scenarios/basic_strike_combat.json tests/snapshots/basic_combat/basic_strike_combat.snap

# Run snapshot test script (compare normal vs battle-only builds)
snapshot-test:
    ./snapshot_test.sh

# Test snapshot determinism (run multiple times and compare)
test-determinism:
    just build-tests
    mkdir -p tests/snapshots/determinism
    ./{{TEST_BUILD_DIR}}/snapshot_generator tests/scenarios/basic_strike_combat.json tests/snapshots/determinism/run1.snap
    ./{{TEST_BUILD_DIR}}/snapshot_generator tests/scenarios/basic_strike_combat.json tests/snapshots/determinism/run2.snap
    ./{{TEST_BUILD_DIR}}/snapshot_generator tests/scenarios/basic_strike_combat.json tests/snapshots/determinism/run3.snap
    diff tests/snapshots/determinism/run1.snap tests/snapshots/determinism/run2.snap
    diff tests/snapshots/determinism/run2.snap tests/snapshots/determinism/run3.snap
    @echo "✓ Snapshots are deterministic!"

# === Run Commands ===

# Run the main interactive simulator
run:
    just build
    ./{{BUILD_DIR}}/main

# Run the standalone battle executable
battle:
    just build
    ./{{BUILD_DIR}}/battle

# Run battle with custom fight JSON file
battle-with-fight fight_json:
    just build
    cp {{fight_json}} battle/sample_fight.json
    ./{{BUILD_DIR}}/battle

# Build only the battle executable
build-battle:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake .. && make battle

# Build only the battle-agent executable
build-battle-agent:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake .. && make battle-agent

# Run neural network agent on JAW_WORM scenarios (requires LibTorch)
run-neural-agent:
    LIBTORCH_PATH=~/Downloads/libtorch just build-with-libtorch
    DYLD_LIBRARY_PATH=~/Downloads/libtorch/lib ./{{BUILD_DIR}}/battle-agent

# Test neural network agent on first 10 JAW_WORM scenarios (requires LibTorch)
test-neural-agent-quick:
    LIBTORCH_PATH=~/Downloads/libtorch just build-with-libtorch
    DYLD_LIBRARY_PATH=~/Downloads/libtorch/lib ./{{BUILD_DIR}}/battle-agent 2>/dev/null | head -50

# Compare agent performance by running both SimpleAgent and NeuralNetAgent
compare-agents:
    @echo "Building both agents..."
    just build
    LIBTORCH_PATH=~/Downloads/libtorch just build-with-libtorch
    @echo "Running SimpleAgent (first 5 scenarios)..."
    ./{{BUILD_DIR}}/battle-agent 2>/dev/null | head -25 | grep -E "(AGENT:|Initial State:|Running agent)"
    @echo "Running NeuralNetAgent (first 5 scenarios)..."
    DYLD_LIBRARY_PATH=~/Downloads/libtorch/lib ./{{BUILD_DIR}}/battle-agent 2>/dev/null | head -25 | grep -E "(AGENT:|Initial State:|Running agent)"

# Validate fight JSON syntax
validate-fight-json fight_json:
    @echo "Validating JSON syntax for {{fight_json}}..."
    @python3 -m json.tool {{fight_json}} > /dev/null && echo "✓ Valid JSON syntax" || echo "✗ Invalid JSON syntax"

# Run simulator with save file replay
run-save savefile actionfile:
    just build
    ./{{BUILD_DIR}}/test save {{savefile}} {{actionfile}}

# Run replay from seed and ascension
run-replay seed ascension actionfile:
    just build
    ./{{BUILD_DIR}}/test replay {{seed}} {{ascension}} {{actionfile}}

# === Development Commands ===

# Check project structure and files
check:
    @echo "Project structure:"
    @find . -name "*.cpp" -o -name "*.h" | head -20
    @echo "\nBuild files:"
    @ls -la {{BUILD_DIR}} 2>/dev/null || echo "Build directory not found"
    @echo "\nExecutables:"
    @ls -la {{BUILD_DIR}}/{main,test,small-test,battle} 2>/dev/null || echo "Some executables not found"
    @echo "\nTest files:"
    @ls -la {{TEST_BUILD_DIR}} 2>/dev/null || echo "Test build directory not found"

# Show project statistics
stats:
    @echo "Lines of code:"
    @find src include -name "*.cpp" -o -name "*.h" | xargs wc -l | tail -1
    @echo "\nFile counts:"
    @echo "Headers: $(find include -name "*.h" | wc -l)"
    @echo "Sources: $(find src -name "*.cpp" | wc -l)"
    @echo "Apps: $(find apps -name "*.cpp" | wc -l)"

# Format code (if clang-format is available)
format:
    find src include apps tests/utils -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# === Utility Commands ===

# Show all available executables and their purposes
executables:
    @echo "Available executables:"
    @echo "  main         - Interactive console simulator for manual gameplay"
    @echo "  test         - Comprehensive test runner with multiple commands"
    @echo "  small-test   - Minimal test executable for quick testing"
    @echo "  battle       - Standalone battle context simulator with SimpleAgent"
    @echo "  battle-agent - Battle simulation with agents (SimpleAgent2, NeuralNetAgent if LibTorch available)"
    @echo ""
    @echo "Usage examples:"
    @echo "  just run            # Run interactive simulator"
    @echo "  just battle         # Run standalone battle simulation"
    @echo "  just test-small     # Run minimal test"
    @echo "  just test agent_mt 1 1 0 1984 1 true  # Run AI agent test"

# Show available test commands
test-help:
    just build
    @echo "Available test commands:"
    @echo "  agent_mt <threads> <depth> <ascension> <seed> <count> <print>"
    @echo "  simple_agent_mt <threads> <seed> <count> [print]"
    @echo "  neural_agent_mt <threads> <seed> <count> [print]   # Requires LibTorch"
    @echo "  mcts_save <savefile> <simulations>"
    @echo "  save <savefile> <actionfile>"
    @echo "  replay <seed> <ascension> <actionfile>"

# Clean all build artifacts and temporary files
clean-all: clean
    find . -name "*.o" -delete
    find . -name "*.a" -delete
    find . -name "core" -delete
    rm -f tests/snapshots/determinism/*.snap

# Create a new combat scenario template
new-scenario name:
    @echo '{\n    "name": "{{name}}",\n    "description": "Description here",\n    "seed": 12345,\n    "ascension": 0,\n    "floor": 1,\n    "initial_state": {\n        "player_hp": 80,\n        "player_max_hp": 80,\n        "character_class": "IRONCLAD",\n        "encounter": "CULTIST",\n        "deck": ["STRIKE", "STRIKE", "STRIKE", "STRIKE", "STRIKE", "DEFEND", "DEFEND", "DEFEND", "DEFEND", "BASH"],\n        "relics": [],\n        "potions": []\n    },\n    "action_sequence": ["play_card_0", "end_turn"]\n}' > tests/scenarios/{{name}}.json
    @echo "Created tests/scenarios/{{name}}.json"

# === Scenario Generation Commands ===

# Generate scenario objects from run data (analysis only)
generate-scenarios-analysis input_dir:
    uv run data/scenarios-from-runs.py {{input_dir}}

# Generate scenario files from run data
generate-scenarios input_dir output_dir:
    uv run data/scenarios-from-runs.py {{input_dir}} {{output_dir}}

# Generate scenarios from ironclad runs to battle/generated_scenarios/
generate-scenarios-ironclad:
    uv run data/scenarios-from-runs.py data/2019-runs-json-ironclad/ battle/generated_scenarios/

# Generate scenarios from ironclad runs to battle/generated_scenarios/ (force delete existing files)
generate-scenarios-ironclad-force:
    uv run data/scenarios-from-runs.py data/2019-runs-json-ironclad/ battle/generated_scenarios/ --force

# Example recipes for common development workflows
example-workflow:
    @echo "Example development workflow:"
    @echo "1. just clean-all    # Clean everything"
    @echo "2. just build        # Build main project"
    @echo "3. just build-tests  # Build test infrastructure"
    @echo "4. just snapshot-basic # Generate basic snapshot"
    @echo "5. just test-determinism # Verify determinism"
    @echo "6. just run          # Run interactive simulator"
