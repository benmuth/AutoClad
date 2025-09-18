"""
Convert CommunicationMod JSON to feature vectors by reusing data_parser.py components.
"""

import logging
from typing import Dict, List, Optional

# Import shared functions from data_parser.py
from .data_parser import create_feature_vector, extract_hand_cards


class StateConverter:
    """Converts CommunicationMod JSON to neural network feature vectors using shared data_parser logic"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def convert_to_features(self, game_state: Dict) -> List[float]:
        """
        Convert CommunicationMod JSON to 15-element feature vector using data_parser.py logic.

        Args:
            game_state: CommunicationMod JSON game state

        Returns:
            List of 15 features matching data_parser.py exactly
        """
        try:
            # Validate that we're in combat
            if not self._is_in_combat(game_state):
                self.logger.error("StateConverter only works for combat states")
                raise ValueError("StateConverter only works for combat states")

            # Convert CommunicationMod JSON to data_parser.py expected format
            parsed_state = self._convert_to_parser_format(game_state)

            # Use shared create_feature_vector function for exact compatibility
            features = create_feature_vector(parsed_state)

            self.logger.debug(f"Converted features: {features}")
            return features

        except Exception as e:
            self.logger.error(f"Feature conversion failed: {e}")
            raise RuntimeError(f"Feature conversion failed: {e}")

    def _is_in_combat(self, game_state: Dict) -> bool:
        """Check if we're in a combat state"""
        return (
            'combat_state' in game_state and
            game_state.get('ready_for_command', False) and
            game_state.get('in_game', False)
        )

    def _convert_to_parser_format(self, game_state: Dict) -> Dict:
        """
        Convert CommunicationMod JSON to the format expected by data_parser.py.

        CommunicationMod format → data_parser.py format:
        combat_state.turn → turn
        combat_state.player.current_hp → health
        combat_state.player.max_hp → maxhealth
        etc.
        """
        combat_state = game_state.get('combat_state', {})
        player = combat_state.get('player', {})

        # Convert to data_parser expected format
        parsed_state = {
            'turn': combat_state.get('turn', 0),
            'health': player.get('current_hp', 0),
            'maxhealth': player.get('max_hp', 100),
            'energy': player.get('energy', 0),
            'block': player.get('block', 0),
            'enemy0_hp': self._get_enemy_hp(combat_state),
            'draw_size': len(combat_state.get('draw_pile', [])),
            'discard_size': len(combat_state.get('discard_pile', [])),
            'exhaust_size': len(combat_state.get('exhaust_pile', [])),
        }

        # Convert hand cards to data_parser format
        self._add_hand_cards_to_state(parsed_state, combat_state)

        # Convert potions to data_parser format
        self._add_potions_to_state(parsed_state, game_state.get('potions', []))

        return parsed_state

    def _get_enemy_hp(self, combat_state: Dict) -> int:
        """Extract HP of first enemy (Jaw Worm)"""
        monsters = combat_state.get('monsters', [])
        if not monsters:
            self.logger.warning("No monsters found in combat state")
            return 0

        enemy0_hp = monsters[0].get('current_hp', 0)
        self.logger.debug(f"Enemy 0 HP: {enemy0_hp}")
        return enemy0_hp

    def _add_hand_cards_to_state(self, parsed_state: Dict, combat_state: Dict) -> None:
        """
        Convert CommunicationMod hand format to data_parser format.

        CommunicationMod: combat_state.hand = [card_objects...]
        data_parser: hand_size=N, hand_card0=id, hand_card1=id, etc.
        """
        hand = combat_state.get('hand', [])
        parsed_state['hand_size'] = len(hand)

        for i, card in enumerate(hand):
            if i < 5:  # data_parser expects max 5 cards
                card_id = self._convert_card_to_numeric_id(card)
                parsed_state[f'hand_card{i}'] = card_id

        self.logger.debug(f"Hand: size={len(hand)}, cards={[parsed_state.get(f'hand_card{i}', -1) for i in range(5)]}")

    def _add_potions_to_state(self, parsed_state: Dict, potions: List[Dict]) -> None:
        """
        Convert CommunicationMod potion format to data_parser format.

        CommunicationMod: potions = [potion_objects...]
        data_parser: potion0=id, potion1=id, potion2=id (1 = empty)
        """
        for i in range(3):  # data_parser expects exactly 3 potion slots
            if i < len(potions):
                potion = potions[i]
                potion_id = potion.get('id', '')
                # Empty slots have id "Potion Slot", convert to 1 for data_parser
                if potion_id == "Potion Slot":
                    parsed_state[f'potion{i}'] = 1  # 1 means empty slot in data_parser
                else:
                    # Convert potion ID to numeric (simplified for now)
                    parsed_state[f'potion{i}'] = abs(hash(potion_id)) % 100
            else:
                parsed_state[f'potion{i}'] = 1  # Empty slot

    def _convert_card_to_numeric_id(self, card: Dict) -> int:
        """
        Convert CommunicationMod card to numeric ID.

        TODO: This should match exactly what was used during training data generation.
        For now, using a simple hash approach, but this needs to be replaced with
        the exact mapping from the training data.
        """
        card_id_str = card.get('id', '')
        card_name = card.get('name', '')

        if not card_id_str:
            self.logger.error(f"Card missing ID: {card}")
            raise ValueError(f"Card missing ID: {card}")

        # Simple hash-based conversion (temporary solution)
        # This should be replaced with the exact mapping from training data
        numeric_id = abs(hash(card_id_str)) % 1000

        self.logger.debug(f"Card '{card_name}' (ID: {card_id_str}) -> numeric ID: {numeric_id}")
        return numeric_id