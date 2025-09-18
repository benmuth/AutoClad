"""
Neural network model loading and prediction interface.
Reuses components from main.py to ensure exact compatibility.
"""

import torch
import logging
from pathlib import Path
from typing import Tuple, Optional

# Import shared components from main.py
from main import CardGameNet, make_prediction


class NeuralModel:
    """Model loader and prediction interface using shared main.py components"""

    def __init__(
        self,
        model_path: str = "jaw_worm_model.pth",
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.device = None
        self.input_size = None

        # Load model or fail immediately
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load trained model and scaler or fail with detailed error"""
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Device selection (matching main.py logic exactly)
            # TODO: make GPU device run faster than CPU device (don't use for now)
            # if torch.backends.mps.is_available():
            if False:
                self.device = torch.device("mps")
                self.logger.info(f"Using MPS device: {self.device}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("MPS device not found. Using CPU.")

            # Load model checkpoint
            self.logger.info(f"Loading model from: {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Validate checkpoint format
            required_keys = ["model_state_dict", "scaler", "input_size"]
            for key in required_keys:
                if key not in checkpoint:
                    self.logger.error(f"Model checkpoint missing '{key}'")
                    raise ValueError(f"Invalid model checkpoint format: missing {key}")

            # Extract components
            self.input_size = checkpoint["input_size"]
            self.scaler = checkpoint["scaler"]

            # Initialize model using shared CardGameNet class
            self.model = CardGameNet(self.input_size).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.logger.info(
                f"Model loaded successfully. Input size: {self.input_size}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, game_state_features: list) -> Tuple[int, list]:
        """
        Make a prediction for a game state using shared make_prediction function.

        Args:
            game_state_features: List of 15 feature values matching data_parser.py format

        Returns:
            Tuple of (predicted_action, probabilities)
            predicted_action: 0-4 for hand positions, 5 for end turn
            probabilities: confidence scores for each action
        """
        try:
            # Validate input size
            if len(game_state_features) != self.input_size:
                self.logger.error(
                    f"Feature vector size mismatch. Expected {self.input_size}, got {len(game_state_features)}"
                )
                raise ValueError(
                    f"Feature vector size mismatch. Expected {self.input_size}, got {len(game_state_features)}"
                )

            # Log features before and after normalization for debugging
            self.logger.debug(f"Raw features: {game_state_features}")

            # Use shared make_prediction function for exact compatibility
            predicted_action, probabilities = make_prediction(
                self.model, game_state_features, self.scaler, self.device
            )

            # Log normalized features
            normalized = self.scaler.transform([game_state_features])[0]
            self.logger.debug(f"Normalized features: {normalized.tolist()}")

            self.logger.debug(
                f"Prediction: {predicted_action}, Probabilities: {probabilities}"
            )

            return predicted_action, probabilities.tolist()

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def get_action_name(self, action: int) -> str:
        """Convert action number to human-readable name"""
        # Import here to avoid circular dependency
        from data_parser import get_card_id_to_class_mapping, get_card_names_mapping

        if action == 42:
            return "End Turn"

        # Get card mappings
        class_to_card = {v: k for k, v in get_card_id_to_class_mapping().items()}
        card_names = get_card_names_mapping()

        if action in class_to_card:
            card_id = class_to_card[action]
            card_name = card_names.get(card_id, f"CardId_{card_id}")
            return f"Play {card_name}"

        return f"Unknown Action {action}"
