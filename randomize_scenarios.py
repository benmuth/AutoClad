#!/usr/bin/env python3
"""
Generate randomized scenarios based on existing ones.
Focuses on JAW_WORM encounters for now.
"""

import json
import os
import random
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any


def get_cards_for_model():
    """Get list of cards that the neural model expects (from training data)"""
    # Cards from the model's card ID mapping
    return [
        "STRIKE", "DEFEND", "BASH",  # Basic
        "ANGER", "CLEAVE", "WARCRY", "FLEX", "IRON_WAVE", "BODY_SLAM", "TRUE_GRIT",
        "SHRUG_IT_OFF", "CLASH", "THUNDERCLAP", "POMMEL_STRIKE", "TWIN_STRIKE",
        "CLOTHESLINE", "ARMAMENTS", "HAVOC", "HEADBUTT", "WILD_STRIKE", "HEAVY_BLADE",
        "PERFECTED_STRIKE", "SWORD_BOOMERANG",  # Common
        "EVOLVE", "UPPERCUT", "GHOSTLY_ARMOR", "FIRE_BREATHING", "DROPKICK", "CARNAGE",
        "BLOODLETTING", "RUPTURE", "SECOND_WIND", "SEARING_BLOW", "BATTLE_TRANCE",
        "SENTINEL", "ENTRENCH", "RAGE", "FEEL_NO_PAIN", "DISARM", "SEEING_RED",
        "DARK_EMBRACE", "COMBUST", "WHIRLWIND", "SEVER_SOUL", "RAMPAGE", "SHOCKWAVE",
        "METALLICIZE", "BURNING_PACT", "PUMMEL", "FLAME_BARRIER", "BLOOD_FOR_BLOOD",
        "INTIMIDATE", "HEMOKINESIS", "RECKLESS_CHARGE", "INFERNAL_BLADE", "DUAL_WIELD",
        "POWER_THROUGH", "INFLAME", "SPOT_WEAKNESS",  # Uncommon
        "DOUBLE_TAP", "DEMON_FORM", "BLUDGEON", "FEED", "LIMIT_BREAK", "CORRUPTION",
        "BARRICADE", "FIEND_FIRE", "BERSERK", "IMPERVIOUS", "JUGGERNAUT", "BRUTALITY",
        "REAPER", "EXHUME", "OFFERING", "IMMOLATE"  # Rare
    ]


def load_jaw_worm_scenarios(scenarios_dir: str) -> List[Dict[Any, Any]]:
    """Load all JAW_WORM scenarios from the directory"""
    scenarios = []
    scenarios_path = Path(scenarios_dir)

    for json_file in scenarios_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                scenario = json.load(f)

            # Check if it's a JAW_WORM encounter
            if (scenario.get("initial_state", {}).get("encounter") == "JAW_WORM"):
                scenarios.append(scenario)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load {json_file}: {e}")

    print(f"Loaded {len(scenarios)} JAW_WORM scenarios")
    return scenarios


def randomize_scenario(base_scenario: Dict[Any, Any], available_cards: List[str]) -> Dict[Any, Any]:
    """Create a randomized version of a scenario"""
    scenario = json.loads(json.dumps(base_scenario))  # Deep copy

    # Generate new random seed
    scenario["seed"] = random.randint(1, 2147483647)

    initial_state = scenario["initial_state"]

    # Randomize player HP (10 to max_hp)
    max_hp = initial_state["player_max_hp"]
    min_hp = max(10, max_hp // 4)  # At least 10, or 25% of max HP
    initial_state["player_hp"] = random.randint(min_hp, max_hp)

    # Randomize deck - add/remove 1-3 cards
    deck = initial_state["deck"].copy()
    num_changes = random.randint(1, 3)

    for _ in range(num_changes):
        if random.choice([True, False]) and len(deck) > 5:  # Remove card (keep at least 5)
            # Prioritize removing basic cards (STRIKE, DEFEND, BASH)
            basic_cards = [card for card in deck if card in ["STRIKE", "DEFEND", "BASH"]]
            non_basic_cards = [card for card in deck if card not in ["STRIKE", "DEFEND", "BASH"]]

            if basic_cards and len(deck) > 10:  # Only remove basics if we have plenty of cards
                deck.remove(random.choice(basic_cards))
            elif non_basic_cards:
                deck.remove(random.choice(non_basic_cards))
            else:
                deck.remove(random.choice(deck))
        else:  # Add card
            new_card = random.choice(available_cards)
            deck.append(new_card)

    initial_state["deck"] = deck

    # Update name and description
    scenario["name"] = f"Randomized JAW_WORM Scenario {random.randint(1000, 9999)}"
    scenario["description"] = f"Randomized scenario based on original data (seed: {scenario['seed']})"

    return scenario


def main():
    parser = argparse.ArgumentParser(description="Generate randomized battle scenarios")
    parser.add_argument(
        "--encounter",
        default="JAW_WORM",
        help="Filter scenarios by encounter type (default: JAW_WORM)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of randomized scenarios to generate (default: 50)"
    )
    parser.add_argument(
        "--input-dir",
        default="battle/generated_scenarios",
        help="Directory containing input scenarios"
    )
    parser.add_argument(
        "--output-dir",
        default="battle/randomized_scenarios",
        help="Directory to output randomized scenarios"
    )

    args = parser.parse_args()

    # Create/clear output directory
    output_path = Path(args.output_dir)
    if output_path.exists():
        response = input(f"Directory '{args.output_dir}' exists and will be deleted. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return 1
        shutil.rmtree(output_path)
        print(f"Removed existing directory: {args.output_dir}")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")

    # Load scenarios
    if args.encounter == "JAW_WORM":
        scenarios = load_jaw_worm_scenarios(args.input_dir)
    else:
        print(f"Error: Only JAW_WORM encounters are supported for now")
        return 1

    if not scenarios:
        print("No scenarios found!")
        return 1

    # Get available cards
    available_cards = get_cards_for_model()
    print(f"Using {len(available_cards)} available cards for randomization")

    # Generate randomized scenarios for each matching encounter
    print(f"Generating {args.count} randomized scenarios for each of {len(scenarios)} base scenarios...")

    total_generated = 0
    for scenario_idx, base_scenario in enumerate(scenarios):
        for i in range(args.count):
            # Randomize the base scenario
            randomized = randomize_scenario(base_scenario, available_cards)

            # Save it with a unique name based on original and index
            original_name = Path(base_scenario.get("name", f"scenario_{scenario_idx}"))
            output_file = output_path / f"randomized_jaw_worm_{scenario_idx:04d}_{i:04d}.json"

            with open(output_file, 'w') as f:
                json.dump(randomized, f, indent=2)

            total_generated += 1

    print(f"Generated {total_generated} total randomized scenarios ({args.count} per base scenario) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())