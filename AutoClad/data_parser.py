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
    
    # Potions (simplified to count of non-empty slots)
    potion_count = 0
    for i in range(3):
        potion_key = f'potion{i}'
        if state.get(potion_key, 1) != 1:  # 1 means empty slot
            potion_count += 1
    features.append(potion_count)
    
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

def parse_battle_file(file_path: str) -> List[Tuple[List[float], int]]:
    """Parse a single battle file and extract state-action pairs."""
    state_action_pairs = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find all state lines
    state_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("turn:"):
            state_lines.append(line)
    
    # Create state-action pairs by comparing consecutive states
    for i in range(len(state_lines) - 1):
        state_before = parse_state_line(state_lines[i])
        state_after = parse_state_line(state_lines[i + 1])
        
        if state_before and state_after:
            # Extract features from before state
            features = create_feature_vector(state_before)
            
            # Infer action that led to after state
            action = infer_action_from_states(state_before, state_after)
            
            if action is not None:
                state_action_pairs.append((features, action))
    
    return state_action_pairs

def load_jaw_worm_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all Jaw Worm battle data for training."""
    # Find all single Jaw Worm battle files (not Horde)
    pattern = os.path.join(data_dir, "SimpleAgent_vs_Jaw Worm_*.snap")
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