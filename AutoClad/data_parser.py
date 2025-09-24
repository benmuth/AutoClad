import os
import re
import glob
import numpy as np
from typing import List, Tuple, Dict, Optional

# 42 card types + 1 end turn action
num_actions = 43


def parse_state_line(line: str) -> Optional[Dict]:
    """Parse a condensed state line into a dictionary of features."""
    if not line.startswith("turn:"):
        return None

    # Extract all key-value pairs from the state line
    state = {}

    # Basic pattern matching for key:value pairs
    pattern = r"(\w+):([^,]+)"
    matches = re.findall(pattern, line)

    for key, value in matches:
        # Convert to appropriate types
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            state[key] = int(value)
        elif value.replace(".", "").isdigit():
            state[key] = float(value)
        else:
            state[key] = value

    return state


def extract_hand_cards(state: Dict) -> List[int]:
    """Extract hand card IDs from parsed state."""
    hand_cards = []
    hand_size = state.get("hand_size", 0)

    for i in range(5):  # Always extract 5 slots (pad with -1 if needed)
        card_key = f"hand_card{i}"
        if i < hand_size and card_key in state:
            hand_cards.append(state[card_key])
        else:
            hand_cards.append(-1)  # Empty slot

    return hand_cards


def encode_hand_cards_count(hand_cards: List[int]) -> List[float]:
    """Convert hand card IDs to count encoding.

    Args:
        hand_cards: List of 5 card IDs (-1 for empty slots)

    Returns:
        List of num_actions features (count of each card type)
    """
    # All unique card IDs found in training data + empty slot (-1)
    CARD_IDS = [
        -1,
        11,
        14,
        25,
        27,
        42,
        52,
        57,
        67,
        69,
        73,
        104,
        112,
        141,
        147,
        148,
        149,
        154,
        170,
        179,
        184,
        187,
        193,
        198,
        202,
        222,
        234,
        244,
        249,
        258,
        261,
        266,
        289,
        297,
        300,
        301,
        311,
        321,
        329,
        339,
        347,
        349,
        362,
    ]

    # Create mapping from card ID to index
    card_to_idx = {card_id: i for i, card_id in enumerate(CARD_IDS)}

    # Count each card type
    card_counts = [0.0] * len(CARD_IDS)

    for card_id in hand_cards:
        if card_id in card_to_idx:
            card_counts[card_to_idx[card_id]] += 1.0
        else:
            # Unknown card - count as empty slot
            card_counts[card_to_idx[-1]] += 1.0

    return card_counts


def create_feature_vector(state: Dict) -> List[float]:
    """Convert game state to feature vector for neural network."""
    features = []

    # Basic game state features (remove max health and pile counts)
    features.extend(
        [
            state.get("turn", 0),
            state.get("health", 0),
            state.get("energy", 0),
            state.get("block", 0),
            state.get("enemy0_hp", 0),  # Jaw Worm HP
        ]
    )

    # Count encoded hand cards (num_actions features for card type counts)
    hand_cards = extract_hand_cards(state)
    hand_counts = encode_hand_cards_count(hand_cards)
    features.extend(hand_counts)

    # Potions (simplified to count of non-empty slots) - COMMENTED OUT FOR NOW
    # potion_count = 0
    # for i in range(3):
    #     potion_key = f'potion{i}'
    #     if state.get(potion_key, 1) != 1:  # 1 means empty slot
    #         potion_count += 1
    # features.append(potion_count)
    features.append(0)  # Placeholder for potion count

    return features


def parse_snap_file_format(lines: List[str]) -> List[Tuple[Dict, Optional[str]]]:
    """Parse actual snap file format and extract game states with their preceding actions."""
    battle_contexts = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for BattleContext sections
        if line.startswith("BattleContext:"):
            context = parse_battle_context_section(lines, i)
            if context:
                # Look for the Last Player Action that precedes this context
                last_action = find_last_player_action_before_context(lines, i)
                battle_contexts.append((context, last_action))
        i += 1

    return battle_contexts


def find_last_player_action_before_context(
    lines: List[str], context_start_idx: int
) -> Optional[str]:
    """Find the 'Last Player Action' line that precedes the given BattleContext."""
    # Search backwards from the context start to find the last action
    for i in range(context_start_idx - 1, max(0, context_start_idx - 20), -1):
        line = lines[i].strip()
        if line.startswith("Last Player Action:"):
            # Extract just the action part, removing "Last Player Action: " prefix
            return line[len("Last Player Action: ") :]
    return None


def parse_last_player_action(action_str: str) -> Optional[int]:
    """Parse a Last Player Action string to determine the action taken.

    Args:
        action_str: String like "(Defend,104,1,1)_target_0" or "(end_turn)"

    Returns:
        Action index: 0-4 for hand positions, 5 for END_TURN, None if invalid
    """
    if not action_str:
        return None

    if action_str == "(end_turn)":
        return 5

    # Extract card info from format like "(Defend,104,1,1)_target_0"
    # Last Player Action format: (CardName,cardId,cost,costForTurn)_target_0
    import re

    match = re.match(r"\(([^,]+),(\d+),(\d+),(\d+)\)_target_\d+", action_str)
    if not match:
        return None

    card_name, card_id, cost, cost_for_turn = match.groups()

    return {
        "card_name": card_name,
        "card_id": int(card_id),  # This is the actual CardId enum value
        "cost": int(cost),
        "cost_for_turn": int(cost_for_turn),
    }


def parse_battle_context_section(lines: List[str], start_idx: int) -> Optional[Dict]:
    """Parse a complete BattleContext section from snap file."""
    import re

    state = {
        "turn": 0,
        "health": 0,
        "maxhealth": 100,
        "energy": 0,
        "block": 0,
        "enemy0_hp": 0,
        "hand_size": 0,
        "draw_size": 0,
        "discard_size": 0,
        "exhaust_size": 0,
    }

    # Find the opening bracket
    i = start_idx
    while i < len(lines) and "{" not in lines[i]:
        i += 1

    if i >= len(lines):
        return None

    # Count brackets to find the matching closing bracket
    bracket_count = 0
    started = False

    while i < len(lines):
        line = lines[i].strip()

        # Count brackets
        bracket_count += line.count("{")
        bracket_count -= line.count("}")

        if bracket_count > 0:
            started = True

        if started:
            # Extract turn information
            if "turn:" in line:
                turn_match = re.search(r"turn:\s*(\d+)", line)
                if turn_match:
                    state["turn"] = int(turn_match.group(1))

            # Extract player HP/energy/block - hp:(102/145) energy:(3/3) block:(0)
            if "hp:(" in line and "energy:(" in line:
                hp_match = re.search(r"hp:\((\d+)/(\d+)\)", line)
                if hp_match:
                    state["health"] = int(hp_match.group(1))
                    state["maxhealth"] = int(hp_match.group(2))

                energy_match = re.search(r"energy:\((\d+)/(\d+)\)", line)
                if energy_match:
                    state["energy"] = int(energy_match.group(1))

                block_match = re.search(r"block:\((\d+)\)", line)
                if block_match:
                    state["block"] = int(block_match.group(1))

            # Extract monster HP - {0 JAW_WORM hp:(43/43)
            if "JAW_WORM" in line and "hp:(" in line:
                hp_match = re.search(r"hp:\((\d+)/\d+\)", line)
                if hp_match:
                    state["enemy0_hp"] = int(hp_match.group(1))

            # Extract hand information - hand: 5 { (Strike,3,1,1), (Defend,7,1,1), ... }
            if "hand:" in line:
                hand_data = parse_card_pile_line(line)
                state["hand_size"] = len(hand_data["cards"])
                # Add individual hand cards
                for idx, card in enumerate(hand_data["cards"][:5]):  # Max 5 cards
                    state[f"hand_card{idx}"] = card["id"]

            # Extract draw pile size
            if "drawPile:" in line:
                pile_data = parse_card_pile_line(line)
                state["draw_size"] = len(pile_data["cards"])

            # Extract discard pile size
            if "discardPile:" in line:
                pile_data = parse_card_pile_line(line)
                state["discard_size"] = len(pile_data["cards"])

            # Extract exhaust pile size
            if "exhaustPile:" in line:
                pile_data = parse_card_pile_line(line)
                state["exhaust_size"] = len(pile_data["cards"])

            # Extract potions - look for EMPTY_POTION_SLOT vs actual potions - COMMENTED OUT FOR NOW
            # if 'EMPTY_POTION_SLOT' in line:
            #     # Count non-empty potions (simplified)
            #     empty_count = line.count('EMPTY_POTION_SLOT')
            #     total_slots = 3  # Assume 3 potion slots
            #     state['potion0'] = 1 if empty_count >= 1 else 2  # 1 = empty, 2 = has potion
            #     state['potion1'] = 1 if empty_count >= 2 else 2
            #     state['potion2'] = 1 if empty_count >= 3 else 2

        # Stop when we've closed all brackets
        if started and bracket_count == 0:
            break

        i += 1

    return state


def parse_card_pile_line(line: str) -> Dict:
    """Parse a line like: hand: 5 { (Strike,3,321,1,1), (Defend,7,104,1,1) }"""
    import re

    # Extract cards in parentheses format (CardName,uniqueId,cardId,cost,costForTurn)
    card_pattern = r"\(([^,]+),(\d+),(\d+),(\d+),(\d+)\)"
    matches = re.findall(card_pattern, line)

    cards = []
    for match in matches:
        name, unique_id, card_id, cost, cost_for_turn = match
        cards.append(
            {
                "name": name,
                "id": int(card_id),  # Use the actual CardId enum value for training
                "unique_id": int(unique_id),  # Keep uniqueId for reference
                "cost": int(cost),
                "cost_for_turn": int(cost_for_turn),
            }
        )

    return {"cards": cards}


def parse_battle_file(file_path: str) -> List[Tuple[List[float], int]]:
    """Parse a single battle file and extract state-action pairs."""
    state_action_pairs = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse actual snap file format - extract BattleContext sections with actions
    battle_contexts_with_actions = parse_snap_file_format(lines)

    # Create state-action pairs: (starting_state, action_taken)
    # The pattern is: state_A -> action_X -> state_B (with "Last Player Action: X")
    # We want training pair: (state_A, action_X)
    for i in range(len(battle_contexts_with_actions) - 1):
        state_before, _ = battle_contexts_with_actions[i]  # Starting state
        state_after, last_action_str = battle_contexts_with_actions[
            i + 1
        ]  # Result state with action

        if state_before and state_after and last_action_str:
            # Extract features from the STARTING state (where decision was made)
            features = create_feature_vector(state_before)

            # Parse the action that was taken from the starting state
            action = convert_last_action_to_action_index(last_action_str, state_before)

            if action is not None:
                state_action_pairs.append((features, action))

    return state_action_pairs


def get_card_id_to_class_mapping():
    """Create mapping from CardId to class index for neural network."""
    # All unique card IDs found in training data (excluding -1 for empty slots)
    CARD_IDS = [
        11,
        14,
        25,
        27,
        42,
        52,
        57,
        67,
        69,
        73,
        104,
        112,
        141,
        147,
        148,
        149,
        154,
        170,
        179,
        184,
        187,
        193,
        198,
        202,
        222,
        234,
        244,
        249,
        258,
        261,
        266,
        289,
        297,
        300,
        301,
        311,
        321,
        329,
        339,
        347,
        349,
        362,
    ]

    # Create mapping: CardId -> class index (0-num_actions-2 for cards, num_actions-1 for end turn)
    card_to_class = {card_id: i for i, card_id in enumerate(CARD_IDS)}

    return card_to_class


def convert_last_action_to_action_index(
    action_str: str, starting_state: Dict
) -> Optional[int]:
    """Convert Last Player Action string to action index based on CardId.

    Args:
        action_str: The "Last Player Action" string like "(Defend,104,1,1)_target_0"
        starting_state: The state where the action was taken from

    Returns:
        Action index: 0-num_actions-2 for card types, num_actions-1 for END_TURN
    """
    if action_str == "(end_turn)":
        return num_actions - 1  # END_TURN class index

    # Parse the action card details
    action_info = parse_last_player_action(action_str)
    if not action_info or isinstance(action_info, int):
        return None

    played_card_id = action_info["card_id"]

    # Get the card ID to class mapping
    card_to_class = get_card_id_to_class_mapping()

    # Convert CardId to class index
    if played_card_id in card_to_class:
        return card_to_class[played_card_id]
    else:
        # Unknown card ID - skip this training example
        return None


def load_jaw_worm_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all Jaw Worm battle data for training."""
    # Find all single Jaw Worm battle files (not Horde)
    pattern = os.path.join(data_dir, "simpleagent", "SimpleAgent_vs_Jaw Worm_*.snap")
    files = glob.glob(pattern)

    print(f"Found {len(files)} Jaw Worm battle files")

    all_states = []
    all_actions = []

    for i, file_path in enumerate(files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(files)}")

        try:
            pairs = parse_battle_file(file_path)
            for features, action in pairs:
                all_states.append(features)
                all_actions.append(action)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Total state-action pairs extracted: {len(all_states)}")

    # Convert to numpy arrays
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    return states, actions


def get_card_names_mapping():
    """Create mapping from CardId to card name for display purposes."""
    # Map of CardId -> Card Name (simplified names based on enum)
    card_names = {
        11: "ANGER",
        14: "ARMAMENTS",
        25: "BASH",
        27: "BATTLE_TRANCE",
        42: "BODY_SLAM",
        52: "BURNING_PACT",
        57: "CARNAGE",
        67: "CLEAVE",
        69: "CLOTHESLINE",
        73: "COMBUST",
        104: "DEFEND_RED",
        112: "DISARM",
        141: "EXHUME",
        147: "FEED",
        148: "FEEL_NO_PAIN",
        149: "FIEND_FIRE",
        154: "FLAME_BARRIER",
        170: "GHOSTLY_ARMOR",
        179: "HEADBUTT",
        184: "HEMOKINESIS",
        187: "IMMOLATE",
        193: "INFLAME",
        198: "IRON_WAVE",
        202: "JUGGERNAUT",
        222: "METALLICIZE",
        234: "OFFERING",
        244: "PERFECTED_STRIKE",
        249: "POMMEL_STRIKE",
        258: "PUMMEL",
        261: "RAGE",
        266: "REAPER",
        289: "SECOND_WIND",
        297: "SEVER_SOUL",
        300: "SHOCKWAVE",
        301: "SHRUG_IT_OFF",
        311: "SPOT_WEAKNESS",
        321: "STRIKE_RED",
        329: "SWORD_BOOMERANG",
        339: "THUNDERCLAP",
        347: "TWIN_STRIKE",
        349: "UPPERCUT",
        362: "WILD_STRIKE",
    }
    return card_names


def analyze_action_distribution(actions: np.ndarray):
    """Analyze the distribution of actions in the dataset."""
    unique_actions, counts = np.unique(actions, return_counts=True)

    print("\nAction Distribution:")

    # Get mappings
    card_to_class = get_card_id_to_class_mapping()
    class_to_card = {v: k for k, v in card_to_class.items()}  # Reverse mapping
    card_names = get_card_names_mapping()

    for action, count in zip(unique_actions, counts):
        if action == num_actions - 1:
            name = "End Turn"
        elif action in class_to_card:
            card_id = class_to_card[action]
            card_name = card_names.get(card_id, f"CardId_{card_id}")
            name = f"Play {card_name}"
        else:
            name = f"Unknown Action {action}"

        percentage = (count / len(actions)) * 100
        print(f"  {name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Test the parser
    data_dir = "/Users/ben/code/sts-ai/data/agent_battles"

    print("Loading Jaw Worm battle data...")
    states, actions = load_jaw_worm_data(data_dir)

    print(f"\nDataset shape:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Feature vector size: {states.shape[1]}")

    # Analyze action distribution
    analyze_action_distribution(actions)

    # Save processed data
    np.savez("jaw_worm_data.npz", states=states, actions=actions)
    print(f"\nSaved processed data to 'jaw_worm_data.npz'")

    # Show sample data
    print(f"\nSample state features (first 10): {states[0][:10]}")
    print(f"Sample action: {actions[0]}")
