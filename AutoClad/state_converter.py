"""
Convert CommunicationMod JSON to feature vectors by reusing data_parser.py components.
"""

import logging
from typing import Dict, List, Optional

# Import shared functions from data_parser.py
from data_parser import create_feature_vector, extract_hand_cards


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
        available_commands = game_state.get("available_commands", [])
        if "play" not in available_commands:
            return False

        # Also check that we have cards in hand
        inner_game_state = game_state.get("game_state", {})
        combat_state = inner_game_state.get("combat_state", {})
        hand = combat_state.get("hand", [])

        if len(hand) == 0:
            self.logger.warning("Combat state detected but hand is empty - refusing to convert")
            return False

        return True

    def _convert_to_parser_format(self, game_state: Dict) -> Dict:
        """
        Convert CommunicationMod JSON to the format expected by data_parser.py.

        CommunicationMod format → data_parser.py format:
        combat_state.turn → turn
        combat_state.player.current_hp → health
        combat_state.player.max_hp → maxhealth
        etc.
        """
        inner_game_state = game_state.get("game_state", {})
        combat_state = inner_game_state.get("combat_state", {})
        player = combat_state.get("player", {})

        # Convert to data_parser expected format
        parsed_state = {
            "turn": combat_state.get("turn", 0),
            "health": player.get("current_hp", 0),
            "maxhealth": player.get("max_hp", 100),
            "energy": player.get("energy", 0),
            "block": player.get("block", 0),
            "enemy0_hp": self._get_enemy_hp(combat_state),
            "draw_size": len(combat_state.get("draw_pile", [])),
            "discard_size": len(combat_state.get("discard_pile", [])),
            "exhaust_size": len(combat_state.get("exhaust_pile", [])),
        }

        # Convert hand cards to data_parser format
        self._add_hand_cards_to_state(parsed_state, combat_state)

        # Convert potions to data_parser format - COMMENTED OUT FOR NOW
        # self._add_potions_to_state(parsed_state, game_state.get('potions', []))

        return parsed_state

    def _get_enemy_hp(self, combat_state: Dict) -> int:
        """Extract HP of first enemy (Jaw Worm)"""
        monsters = combat_state.get("monsters", [])
        if not monsters:
            self.logger.warning("No monsters found in combat state")
            return 0

        enemy0_hp = monsters[0].get("current_hp", 0)
        self.logger.debug(f"Enemy 0 HP: {enemy0_hp}")
        return enemy0_hp

    def _add_hand_cards_to_state(self, parsed_state: Dict, combat_state: Dict) -> None:
        """
        Convert CommunicationMod hand format to data_parser format.

        CommunicationMod: combat_state.hand = [card_objects...]
        data_parser: hand_size=N, hand_card0=id, hand_card1=id, etc.
        """
        hand = combat_state.get("hand", [])
        parsed_state["hand_size"] = len(hand)

        for i, card in enumerate(hand):
            if i < 5:  # data_parser expects max 5 cards
                card_id = self._convert_card_to_numeric_id(card)
                parsed_state[f"hand_card{i}"] = card_id

        self.logger.debug(
            f"Hand: size={len(hand)}, cards={[parsed_state.get(f'hand_card{i}', -1) for i in range(5)]}"
        )

    # COMMENTED OUT FOR NOW - POTIONS DISABLED
    # def _add_potions_to_state(self, parsed_state: Dict, potions: List[Dict]) -> None:
    #     """
    #     Convert CommunicationMod potion format to data_parser format.

    #     CommunicationMod: potions = [potion_objects...]
    #     data_parser: potion0=id, potion1=id, potion2=id (1 = empty)
    #     """
    #     for i in range(3):  # data_parser expects exactly 3 potion slots
    #         if i < len(potions):
    #             potion = potions[i]
    #             potion_id = potion.get('id', '')
    #             # Empty slots have id "Potion Slot", convert to 1 for data_parser
    #             if potion_id == "Potion Slot":
    #                 parsed_state[f'potion{i}'] = 1  # 1 means empty slot in data_parser
    #             else:
    #                 # Convert potion ID to numeric (simplified for now)
    #                 parsed_state[f'potion{i}'] = abs(hash(potion_id)) % 100
    #         else:
    #             parsed_state[f'potion{i}'] = 1  # Empty slot

    def _convert_card_to_numeric_id(self, card: Dict) -> int:
        """
        Convert CommunicationMod card to numeric ID using actual CardId enum values.

        This mapping matches exactly what was used during training data generation.
        Values are extracted directly from the C++ CardId enum in Cards.h.
        """
        card_id_str = card.get("id", "")
        card_name = card.get("name", "")

        if not card_id_str:
            self.logger.error(f"Card missing ID: {card}")
            raise ValueError(f"Card missing ID: {card}")

        # Map CommunicationMod card IDs to actual CardId enum values from Cards.h
        # These are the exact numeric values from the C++ enum
        card_id_mapping = {
            # Basic Ironclad cards
            "Strike_R": 321,  # STRIKE_RED
            "Defend_R": 104,  # DEFEND_RED
            "Bash": 25,  # BASH
            # Common Ironclad cards
            "Anger": 11,  # ANGER
            "Cleave": 67,  # CLEAVE
            "Warcry": 355,  # WARCRY
            "Flex": 157,  # FLEX
            "Iron_Wave": 198,  # IRON_WAVE
            "Body_Slam": 42,  # BODY_SLAM
            "True_Grit": 345,  # TRUE_GRIT
            "Shrug_It_Off": 301,  # SHRUG_IT_OFF
            "Clash": 65,  # CLASH
            "Thunderclap": 339,  # THUNDERCLAP
            "Pommel_Strike": 249,  # POMMEL_STRIKE
            "Twin_Strike": 347,  # TWIN_STRIKE
            "Clothesline": 69,  # CLOTHESLINE
            "Armaments": 14,  # ARMAMENTS
            "Havoc": 178,  # HAVOC
            "Headbutt": 179,  # HEADBUTT
            "Wild_Strike": 362,  # WILD_STRIKE
            "Heavy_Blade": 181,  # HEAVY_BLADE
            "Perfected_Strike": 244,  # PERFECTED_STRIKE
            "Sword_Boomerang": 329,  # SWORD_BOOMERANG
            # Uncommon Ironclad cards
            "Evolve": 140,  # EVOLVE
            "Uppercut": 349,  # UPPERCUT
            "Ghostly_Armor": 170,  # GHOSTLY_ARMOR
            "Fire_Breathing": 152,  # FIRE_BREATHING
            "Dropkick": 122,  # DROPKICK
            "Carnage": 57,  # CARNAGE
            "Bloodletting": 38,  # BLOODLETTING
            "Rupture": 279,  # RUPTURE
            "Second_Wind": 289,  # SECOND_WIND
            "Searing_Blow": 288,  # SEARING_BLOW
            "Battle_Trance": 27,  # BATTLE_TRANCE
            "Sentinel": 295,  # SENTINEL
            "Entrench": 132,  # ENTRENCH
            "Rage": 261,  # RAGE
            "Feel_No_Pain": 148,  # FEEL_NO_PAIN
            "Disarm": 112,  # DISARM
            "Seeing_Red": 292,  # SEEING_RED
            "Dark_Embrace": 93,  # DARK_EMBRACE
            "Combust": 73,  # COMBUST
            "Whirlwind": 360,  # WHIRLWIND
            "Sever_Soul": 297,  # SEVER_SOUL
            "Rampage": 264,  # RAMPAGE
            "Shockwave": 300,  # SHOCKWAVE
            "Metallicize": 222,  # METALLICIZE
            "Burning_Pact": 52,  # BURNING_PACT
            "Pummel": 258,  # PUMMEL
            "Flame_Barrier": 154,  # FLAME_BARRIER
            "Blood_for_Blood": 39,  # BLOOD_FOR_BLOOD
            "Intimidate": 197,  # INTIMIDATE
            "Hemokinesis": 184,  # HEMOKINESIS
            "Reckless_Charge": 269,  # RECKLESS_CHARGE
            "Infernal_Blade": 191,  # INFERNAL_BLADE
            "Dual_Wield": 124,  # DUAL_WIELD
            "Power_Through": 250,  # POWER_THROUGH
            "Inflame": 193,  # INFLAME
            "Spot_Weakness": 311,  # SPOT_WEAKNESS
            # Rare Ironclad cards
            "Double_Tap": 119,  # DOUBLE_TAP
            "Demon_Form": 107,  # DEMON_FORM
            "Bludgeon": 40,  # BLUDGEON
            "Feed": 147,  # FEED
            "Limit_Break": 208,  # LIMIT_BREAK
            "Corruption": 83,  # CORRUPTION
            "Barricade": 24,  # BARRICADE
            "Fiend_Fire": 149,  # FIEND_FIRE
            "Berserk": 30,  # BERSERK
            "Impervious": 189,  # IMPERVIOUS
            "Juggernaut": 202,  # JUGGERNAUT
            "Brutality": 47,  # BRUTALITY
            "Reaper": 266,  # REAPER
            "Exhume": 141,  # EXHUME
            "Offering": 234,  # OFFERING
            "Immolate": 187,  # IMMOLATE
            # Curses
            "AscendersBane": 15,  # ASCENDERS_BANE
        }

        numeric_id = card_id_mapping.get(card_id_str)

        if numeric_id is None:
            self.logger.error(
                f"Unknown card ID '{card_id_str}' (name: '{card_name}'). Need to add mapping."
            )
            # Fatal error - we cannot proceed with unknown cards
            raise ValueError(
                f"Unknown card ID '{card_id_str}' - add mapping to card_id_mapping dictionary"
            )

        self.logger.debug(
            f"Card '{card_name}' (ID: {card_id_str}) -> numeric ID: {numeric_id}"
        )
        return numeric_id
