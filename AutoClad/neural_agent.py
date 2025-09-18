"""
Neural network agent for CommunicationMod protocol.
Uses shared components to ensure exact compatibility with training.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from neural_model import NeuralModel
from state_converter import StateConverter


class NeuralAgent:
    """Neural network agent implementing CommunicationMod protocol"""

    def __init__(
        self, model_path: str = "jaw_worm_model.pth", log_file: str = "neural_agent.log"
    ):
        # Setup file-based logging (avoid stdout/stderr pollution)
        self.setup_logging(log_file)

        # Initialize components
        try:
            self.neural_model = NeuralModel(model_path, self.logger)
            self.state_converter = StateConverter(self.logger)
            self.logger.info("NeuralAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NeuralAgent: {e}")
            sys.exit(1)

    def setup_logging(self, log_file: str) -> None:
        """Setup file-based logging to avoid stdout pollution"""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Main CommunicationMod protocol loop"""
        try:
            # Send ready signal to CommunicationMod
            print("ready", flush=True)
            self.logger.info("Sent ready signal to CommunicationMod")

            # Main protocol loop
            while True:
                try:
                    # Read game state from stdin
                    line = sys.stdin.readline()
                    if not line:
                        self.logger.info("EOF received, exiting")
                        break

                    # Parse JSON game state
                    game_state = json.loads(line.strip())
                    self.logger.debug(f"Received game state: {game_state.keys()}")

                    # Get action from neural network
                    command = self.get_action(game_state)

                    # Send command to CommunicationMod
                    print(command, flush=True)
                    self.logger.info(f"Sent command: {command}")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {e}")
                    print("END", flush=True)  # Safe fallback
                except Exception as e:
                    self.logger.error(f"Error in protocol loop: {e}")
                    print("END", flush=True)  # Safe fallback

        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
            sys.exit(1)

    def get_action(self, game_state: Dict) -> str:
        """
        Get action from neural network or hand off to human for non-combat decisions.

        Args:
            game_state: CommunicationMod JSON game state

        Returns:
            CommunicationMod command string
        """
        try:
            # Check if we can handle this game state
            if not self.can_handle_state(game_state):
                # Log extensive info and check for missing expected fields
                available_commands = game_state.get("available_commands", [])
                in_game = game_state.get("in_game")
                ready_for_command = game_state.get("ready_for_command")
                room_phase = game_state.get("room_phase")

                # Check for missing expected fields
                # if in_game is None:
                #     self.logger.error("Missing 'in_game' field in game state")
                # if ready_for_command is None:
                #     self.logger.error("Missing 'ready_for_command' field in game state")
                # if room_phase is None:
                #     self.logger.error("Missing 'room_phase' field in game state")
                # if not available_commands:
                #     self.logger.error("Empty or missing 'available_commands' field in game state")

                game_state_inner = game_state.get("game_state", {})
                screen_type = game_state_inner.get("screen_type")
                action_phase = game_state_inner.get("action_phase")

                # if not game_state_inner:
                #     self.logger.error("Missing 'game_state' field in game state")
                # if screen_type is None:
                #     self.logger.error("Missing 'screen_type' in game_state")
                # if action_phase is None:
                #     self.logger.error("Missing 'action_phase' in game_state")

                self.logger.info(
                    f"Non-combat state: in_game={in_game}, ready_for_command={ready_for_command}, "
                    f"room_phase='{room_phase}', screen_type='{screen_type}', action_phase='{action_phase}', "
                    f"available_commands={available_commands}"
                )

                # Wait for combat - let humans handle non-combat decisions
                self.logger.info(
                    "Waiting for combat state - human should handle non-combat decisions"
                )
                return "WAIT 200"  # Wait 200 frames (~3.3 seconds) then check again

            # Convert game state to features
            features = self.state_converter.convert_to_features(game_state)

            # Log the feature vector for debugging
            self.logger.info(f"Feature vector: {features}")

            # Get neural network prediction
            predicted_action, probabilities = self.neural_model.predict(features)

            # Convert prediction to CommunicationMod command
            command = self.convert_action_to_command(predicted_action, game_state)

            # Log decision
            action_name = self.neural_model.get_action_name(predicted_action)
            confidence = probabilities[predicted_action] * 100
            self.logger.info(
                f"Neural decision: {action_name} (confidence: {confidence:.1f}%)"
            )

            # Log all action probabilities for debugging
            action_names = ["Hand 0", "Hand 1", "Hand 2", "Hand 3", "Hand 4", "End Turn"]
            for i, (name, prob) in enumerate(zip(action_names, probabilities)):
                self.logger.info(f"  {name}: {prob*100:.1f}%")

            return command

        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            # Fatal error - exit rather than fallback
            sys.exit(1)

    def can_handle_state(self, game_state: Dict) -> bool:
        """Check if this is a combat state we can handle with neural network"""
        return (
            game_state.get("in_game", False)
            and game_state.get("ready_for_command", False)
            and self.is_card_play_state(game_state)
        )

    def is_card_play_state(self, game_state: Dict) -> bool:
        """Check if we're in a state where we can play cards (not card select screens)"""
        # Check available commands - we only handle PLAY and END commands
        available_commands = game_state.get("available_commands", [])

        # We can handle states where we can play cards or end turn
        can_play = "play" in available_commands
        can_end = "end" in available_commands

        return can_play or can_end

    def convert_action_to_command(self, predicted_action: int, game_state: Dict) -> str:
        """
        Convert neural network prediction to CommunicationMod command.

        Args:
            predicted_action: 0-4 for hand positions, 5 for end turn
            game_state: CommunicationMod game state for validation

        Returns:
            CommunicationMod command string
        """
        if predicted_action == 5:
            return "END"
        elif 0 <= predicted_action <= 4:
            # Validate that hand position exists and is playable
            if self.validate_card_play(predicted_action, game_state):
                # Convert 0-indexed to 1-indexed for CommunicationMod
                return f"PLAY {predicted_action + 1}"
            else:
                self.logger.error(f"Invalid card play at position {predicted_action}")
                # Fatal error - don't fallback to END
                sys.exit(1)
        else:
            self.logger.error(f"Invalid action prediction: {predicted_action}")
            sys.exit(1)

    def validate_card_play(self, hand_position: int, game_state: Dict) -> bool:
        """Validate that the predicted card play is legal"""
        combat_state = game_state.get("combat_state", {})
        hand = combat_state.get("hand", [])

        # Check if hand position exists
        if hand_position >= len(hand):
            self.logger.error(
                f"Hand position {hand_position} out of range (hand size: {len(hand)})"
            )
            return False

        # Check if card is playable
        card = hand[hand_position]
        is_playable = card.get("is_playable", False)

        if not is_playable:
            self.logger.error(
                f"Card at position {hand_position} is not playable: {card.get('name', 'Unknown')}"
            )
            return False

        return True


def main():
    """Entry point for neural agent"""
    agent = NeuralAgent()
    agent.run()


if __name__ == "__main__":
    main()
