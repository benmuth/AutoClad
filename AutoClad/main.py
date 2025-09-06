import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


class CardGameDataset(Dataset):
    """Custom dataset for card game states and actions"""

    def __init__(self, states, actions):
        """
        Args:
            states: numpy array of game states (features)
            actions: numpy array of chosen actions (hand positions 0-4)
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(
            actions
        )  # LongTensor for classification targets

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class CardGameNet(nn.Module):
    """Neural network for card game decision making"""

    def __init__(self, input_size, hidden_size1=256, hidden_size2=128):
        super(CardGameNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization to prevent overfitting
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, 6),  # 5 hand positions + 1 end turn action
            # Note: No softmax here - CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        return self.network(x)


def prepare_data(raw_states, raw_actions):
    """
    Prepare and normalize the data

    Args:
        raw_states: list of game states (your feature engineering goes here)
        raw_actions: list of chosen hand positions (0-4)

    Returns:
        normalized_states, actions, scaler
    """
    # Convert to numpy arrays
    states = np.array(raw_states)
    actions = np.array(raw_actions)

    # Normalize features (important for neural networks)
    scaler = StandardScaler()
    normalized_states = scaler.fit_transform(states)

    return normalized_states, actions, scaler


def load_jaw_worm_data():
    """Load the processed Jaw Worm battle data."""
    import numpy as np

    try:
        # Load the preprocessed data
        data = np.load("jaw_worm_data.npz")
        states = data["states"]
        actions = data["actions"]

        print(f"Loaded {len(states)} state-action pairs from Jaw Worm battles")
        print(f"Feature vector size: {states.shape[1]}")

        return states, actions

    except FileNotFoundError:
        print("jaw_worm_data.npz not found. Please run data_parser.py first.")
        return None, None


def train_model(model, train_loader, val_loader, test_loader, num_epochs=50):
    """Train the neural network and evaluate on train/val/test"""

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    # Enable interactive plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_actions.size(0)
            train_correct += (predicted == batch_actions).sum().item()

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_actions.size(0)
                val_correct += (predicted == batch_actions).sum().item()

        # Test phase (just like validation)
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_states, batch_actions in test_loader:
                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_actions.size(0)
                test_correct += (predicted == batch_actions).sum().item()

        # Store metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        test_acc = 100 * test_correct / test_total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["test_loss"].append(avg_test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print("-" * 50)

        # === Live Plot Update ===
        ax1.clear()
        ax2.clear()

        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"], label="Val Loss")
        ax1.plot(epochs, history["test_loss"], label="Test Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss over Epochs")
        ax1.legend()

        ax2.plot(epochs, history["train_acc"], label="Train Acc")
        ax2.plot(epochs, history["val_acc"], label="Val Acc")
        ax2.plot(epochs, history["test_acc"], label="Test Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy over Epochs")
        ax2.legend()

        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    return history


def make_prediction(model, game_state, scaler):
    """
    Make a prediction for a single game state

    Args:
        model: trained PyTorch model
        game_state: feature vector for current game state
        scaler: fitted StandardScaler from training

    Returns:
        predicted_action: which action to take (0-4: hand positions, 5: end turn)
        probabilities: confidence scores for each action (0-4: hand positions, 5: end turn)
    """
    model.eval()

    # Normalize the input
    normalized_state = scaler.transform([game_state])
    state_tensor = torch.FloatTensor(normalized_state)

    with torch.no_grad():
        outputs = model(state_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_action = torch.max(outputs, 1)

    return predicted_action.item(), probabilities[0].numpy()


# Example usage:
if __name__ == "__main__":
    # Load real Jaw Worm battle data
    print("Loading Jaw Worm battle data...")
    raw_states, raw_actions = load_jaw_worm_data()

    if raw_states is None or raw_actions is None:
        print("Failed to load data. Exiting.")
        exit(1)

    # 2. Prepare data (normalize features)
    states, actions, scaler = prepare_data(raw_states, raw_actions)

    # 3. Create dataset and data loaders
    dataset = CardGameDataset(states, actions)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. Create and train model
    input_size = states.shape[1]  # Size of feature vector from real data
    model = CardGameNet(input_size)

    print(f"Model created with input size: {input_size}")
    print(f"Training on {len(train_dataset)} examples")
    print("Starting training...")

    _ = train_model(model, train_loader, val_loader, test_loader, num_epochs=200)

    # 5. Save the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "input_size": input_size,
        },
        "jaw_worm_model.pth",
    )

    print("Model saved as 'jaw_worm_model.pth'")

    # Export model to TorchScript for C++ usage
    model.eval()
    example_input = torch.randn(1, input_size)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("jaw_worm_model_traced.pt")

    # Save scaler parameters for C++ normalization
    import json

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "input_size": input_size,
    }
    with open("scaler_params.json", "w") as f:
        json.dump(scaler_params, f)

    print("TorchScript model saved as 'jaw_worm_model_traced.pt'")
    print("Scaler parameters saved as 'scaler_params.json'")

    # 6. Example prediction
    example_state = raw_states[0]
    predicted_action, probs = make_prediction(model, example_state, scaler)
    action_names = ["Hand 0", "Hand 1", "Hand 2", "Hand 3", "Hand 4", "End Turn"]
    print(f"\nExample prediction:")
    print(f"Predicted action: {predicted_action} ({action_names[predicted_action]})")
    print(f"Confidence scores:")
    for i, (name, prob) in enumerate(zip(action_names, probs)):
        marker = " ←" if i == predicted_action else ""
        print(f"  {name}: {prob:.1%}{marker}")
    print(f"Actual action was: {raw_actions[0]} ({action_names[raw_actions[0]]})")
