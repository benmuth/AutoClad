import os
import re
import glob
import numpy as np
from typing import List, Tuple, Dict, Optional

def parse_state_line(line: str) -> Optional[Dict]:
    """Parse a condensed state line into a dictionary of features."""
    if not line.startswith("turn:"):
        return None
    
    # Extract all key-value pairs from the state line
    state = {}
    
    # Basic pattern matching for key:value pairs
    pattern = r'(\w+):([^,]+)'
    matches = re.findall(pattern, line)
    
    for key, value in matches:
        # Convert to appropriate types
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            state[key] = int(value)
        elif value.replace('.', '').isdigit():
            state[key] = float(value)
        else:
            state[key] = value
    
    return state

def extract_hand_cards(state: Dict) -> List[int]:
    """Extract hand card IDs from parsed state."""
    hand_cards = []
    hand_size = state.get('hand_size', 0)
    
    for i in range(5):  # Always extract 5 slots (pad with -1 if needed)
        card_key = f'hand_card{i}'
        if i < hand_size and card_key in state:
            hand_cards.append(state[card_key])
        else:
            hand_cards.append(-1)  # Empty slot
    
    return hand_cards

def create_feature_vector(state: Dict) -> List[float]:
    """Convert game state to feature vector for neural network."""
    features = []
    
    # Basic game state features
    features.extend([
        state.get('turn', 0),
        state.get('health', 0),
        state.get('maxhealth', 100),
        state.get('energy', 0),
        state.get('block', 0),
        state.get('enemy0_hp', 0),  # Jaw Worm HP
    ])
    
    # Hand cards (5 cards, pad with -1 if needed)
    hand_cards = extract_hand_cards(state)
    features.extend(hand_cards)
    
    # Draw pile size
    features.append(state.get('draw_size', 0))
    
    # Discard pile size  
    features.append(state.get('discard_size', 0))
    
    # Exhaust pile size
    features.append(state.get('exhaust_size', 0))
    
    # Potions (simplified to count of non-empty slots) - COMMENTED OUT FOR NOW
    # potion_count = 0
    # for i in range(3):
    #     potion_key = f'potion{i}'
    #     if state.get(potion_key, 1) != 1:  # 1 means empty slot
    #         potion_count += 1
    # features.append(potion_count)
    features.append(0)  # Placeholder for potion count
    
    return features

def infer_action_from_states(state_before: Dict, state_after: Dict) -> Optional[int]:
    """Infer which action was taken by comparing before/after states."""
    # Simple heuristic: if a card was played, find which hand position changed
    hand_before = extract_hand_cards(state_before)
    hand_after = extract_hand_cards(state_after)
    
    # Find first difference in hand (assuming single card play)
    for i in range(5):
        if hand_before[i] != hand_after[i]:
            # Card at position i was played
            return i
    
    # If no hand change detected, might be END_TURN action
    # We'll represent END_TURN as action 5 (beyond hand positions 0-4)
    if state_before.get('turn', 0) != state_after.get('turn', 0):
        return 5  # END_TURN
    
    return None

def parse_snap_file_format(lines: List[str]) -> List[Tuple[Dict, Optional[str]]]:
    """Parse actual snap file format and extract game states with their preceding actions."""
    battle_contexts = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for BattleContext sections
        if line.startswith('BattleContext:'):
            context = parse_battle_context_section(lines, i)
            if context:
                # Look for the Last Player Action that precedes this context
                last_action = find_last_player_action_before_context(lines, i)
                battle_contexts.append((context, last_action))
        i += 1

    return battle_contexts

def find_last_player_action_before_context(lines: List[str], context_start_idx: int) -> Optional[str]:
    """Find the 'Last Player Action' line that precedes the given BattleContext."""
    # Search backwards from the context start to find the last action
    for i in range(context_start_idx - 1, max(0, context_start_idx - 20), -1):
        line = lines[i].strip()
        if line.startswith('Last Player Action:'):
            # Extract just the action part, removing "Last Player Action: " prefix
            return line[len('Last Player Action: '):]
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
    match = re.match(r'\(([^,]+),(\d+),(\d+),(\d+)\)_target_\d+', action_str)
    if not match:
        return None

    card_name, card_id, cost, cost_for_turn = match.groups()

    return {
        'card_name': card_name,
        'card_id': int(card_id),  # This is the actual CardId enum value
        'cost': int(cost),
        'cost_for_turn': int(cost_for_turn)
    }

def parse_battle_context_section(lines: List[str], start_idx: int) -> Optional[Dict]:
    """Parse a complete BattleContext section from snap file."""
    import re

    state = {
        'turn': 0,
        'health': 0,
        'maxhealth': 100,
        'energy': 0,
        'block': 0,
        'enemy0_hp': 0,
        'hand_size': 0,
        'draw_size': 0,
        'discard_size': 0,
        'exhaust_size': 0
    }

    # Find the opening bracket
    i = start_idx
    while i < len(lines) and '{' not in lines[i]:
        i += 1

    if i >= len(lines):
        return None

    # Count brackets to find the matching closing bracket
    bracket_count = 0
    started = False

    while i < len(lines):
        line = lines[i].strip()

        # Count brackets
        bracket_count += line.count('{')
        bracket_count -= line.count('}')

        if bracket_count > 0:
            started = True

        if started:
            # Extract turn information
            if 'turn:' in line:
                turn_match = re.search(r'turn:\s*(\d+)', line)
                if turn_match:
                    state['turn'] = int(turn_match.group(1))

            # Extract player HP/energy/block - hp:(102/145) energy:(3/3) block:(0)
            if 'hp:(' in line and 'energy:(' in line:
                hp_match = re.search(r'hp:\((\d+)/(\d+)\)', line)
                if hp_match:
                    state['health'] = int(hp_match.group(1))
                    state['maxhealth'] = int(hp_match.group(2))

                energy_match = re.search(r'energy:\((\d+)/(\d+)\)', line)
                if energy_match:
                    state['energy'] = int(energy_match.group(1))

                block_match = re.search(r'block:\((\d+)\)', line)
                if block_match:
                    state['block'] = int(block_match.group(1))

            # Extract monster HP - {0 JAW_WORM hp:(43/43)
            if 'JAW_WORM' in line and 'hp:(' in line:
                hp_match = re.search(r'hp:\((\d+)/\d+\)', line)
                if hp_match:
                    state['enemy0_hp'] = int(hp_match.group(1))

            # Extract hand information - hand: 5 { (Strike,3,1,1), (Defend,7,1,1), ... }
            if 'hand:' in line:
                hand_data = parse_card_pile_line(line)
                state['hand_size'] = len(hand_data['cards'])
                # Add individual hand cards
                for idx, card in enumerate(hand_data['cards'][:5]):  # Max 5 cards
                    state[f'hand_card{idx}'] = card['id']

            # Extract draw pile size
            if 'drawPile:' in line:
                pile_data = parse_card_pile_line(line)
                state['draw_size'] = len(pile_data['cards'])

            # Extract discard pile size
            if 'discardPile:' in line:
                pile_data = parse_card_pile_line(line)
                state['discard_size'] = len(pile_data['cards'])

            # Extract exhaust pile size
            if 'exhaustPile:' in line:
                pile_data = parse_card_pile_line(line)
                state['exhaust_size'] = len(pile_data['cards'])

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
    card_pattern = r'\(([^,]+),(\d+),(\d+),(\d+),(\d+)\)'
    matches = re.findall(card_pattern, line)

    cards = []
    for match in matches:
        name, unique_id, card_id, cost, cost_for_turn = match
        cards.append({
            'name': name,
            'id': int(card_id),  # Use the actual CardId enum value for training
            'unique_id': int(unique_id),  # Keep uniqueId for reference
            'cost': int(cost),
            'cost_for_turn': int(cost_for_turn)
        })

    return {'cards': cards}

def parse_battle_file(file_path: str) -> List[Tuple[List[float], int]]:
    """Parse a single battle file and extract state-action pairs."""
    state_action_pairs = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse actual snap file format - extract BattleContext sections with actions
    battle_contexts_with_actions = parse_snap_file_format(lines)

    # Create state-action pairs: (starting_state, action_taken)
    # The pattern is: state_A -> action_X -> state_B (with "Last Player Action: X")
    # We want training pair: (state_A, action_X)
    for i in range(len(battle_contexts_with_actions) - 1):
        state_before, _ = battle_contexts_with_actions[i]  # Starting state
        state_after, last_action_str = battle_contexts_with_actions[i + 1]  # Result state with action

        if state_before and state_after and last_action_str:
            # Extract features from the STARTING state (where decision was made)
            features = create_feature_vector(state_before)

            # Parse the action that was taken from the starting state
            action = convert_last_action_to_action_index(last_action_str, state_before)

            if action is not None:
                state_action_pairs.append((features, action))

    return state_action_pairs

def convert_last_action_to_action_index(action_str: str, starting_state: Dict) -> Optional[int]:
    """Convert Last Player Action string to action index (0-5).

    Args:
        action_str: The "Last Player Action" string like "(Defend,104,1,1)_target_0"
        starting_state: The state where the action was taken from

    Returns:
        Action index: 0-4 for hand positions, 5 for END_TURN
    """
    if action_str == "(end_turn)":
        return 5

    # Parse the action card details
    action_info = parse_last_player_action(action_str)
    if not action_info or isinstance(action_info, int):
        return None

    played_card_id = action_info['card_id']
    # Look through the starting state's hand to find any position with this card ID
    for hand_pos in range(5):
        hand_card_key = f'hand_card{hand_pos}'
        if hand_card_key in starting_state:
            if starting_state[hand_card_key] == played_card_id:
                return hand_pos

    # If we can't find a match, return None
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

def analyze_action_distribution(actions: np.ndarray):
    """Analyze the distribution of actions in the dataset."""
    unique_actions, counts = np.unique(actions, return_counts=True)
    
    print("\nAction Distribution:")
    action_names = {
        0: "Play Hand Card 0",
        1: "Play Hand Card 1", 
        2: "Play Hand Card 2",
        3: "Play Hand Card 3",
        4: "Play Hand Card 4",
        5: "End Turn"
    }
    
    for action, count in zip(unique_actions, counts):
        name = action_names.get(action, f"Unknown Action {action}")
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
    np.savez('jaw_worm_data.npz', states=states, actions=actions)
    print(f"\nSaved processed data to 'jaw_worm_data.npz'")
    
    # Show sample data
    print(f"\nSample state features (first 10): {states[0][:10]}")
    print(f"Sample action: {actions[0]}")