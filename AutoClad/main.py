import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CardGameDataset(Dataset):
    """Custom dataset for card game states and actions"""
    
    def __init__(self, states, actions):
        """
        Args:
            states: numpy array of game states (features)
            actions: numpy array of chosen actions (hand positions 0-4)
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)  # LongTensor for classification targets
        
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
            nn.Linear(hidden_size2, 5)  # 5 hand positions
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

def create_feature_vector(player_health, enemy_health, cards_in_hand, active_effects, num_total_cards=75):
    """
    Convert game state to feature vector
    
    Args:
        player_health: int
        enemy_health: int  
        cards_in_hand: list of card IDs [23, 45, 12, 67, 3]
        active_effects: list of active effect IDs
        num_total_cards: total number of possible cards
        
    Returns:
        feature vector as list
    """
    features = []
    
    # Health values
    features.extend([player_health, enemy_health])
    
    # One-hot encode cards in hand (5 cards × 75 possible cards = 375 features)
    hand_encoding = [0] * (5 * num_total_cards)
    for position, card_id in enumerate(cards_in_hand):
        if card_id is not None:  # Handle cases with fewer than 5 cards
            hand_encoding[position * num_total_cards + card_id] = 1
    features.extend(hand_encoding)
    
    # One-hot encode active effects (you'll need to define max_effects)
    # This is a simplified version - you'll need to adapt based on your game
    max_effects = 10  # Adjust based on your game
    effect_encoding = [0] * max_effects
    for effect_id in active_effects[:max_effects]:  # Truncate if too many
        if effect_id < max_effects:
            effect_encoding[effect_id] = 1
    features.extend(effect_encoding)
    
    return features

def train_model(model, train_loader, val_loader, num_epochs=50):
    """Train the neural network"""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Perfect for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_states, batch_actions in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            
            # Backward pass and optimization
            loss.backward()  # This is where the magic happens - automatic gradients!
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_actions.size(0)
            train_correct += (predicted == batch_actions).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # No gradients needed for validation
            for batch_states, batch_actions in val_loader:
                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_actions.size(0)
                val_correct += (predicted == batch_actions).sum().item()
        
        # Print progress
        if epoch % 10 == 0:
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            print(f'Epoch [{epoch}/{num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)

def make_prediction(model, game_state, scaler):
    """
    Make a prediction for a single game state
    
    Args:
        model: trained PyTorch model
        game_state: feature vector for current game state
        scaler: fitted StandardScaler from training
        
    Returns:
        predicted_position: which hand position to play (0-4)
        probabilities: confidence scores for each position
    """
    model.eval()
    
    # Normalize the input
    normalized_state = scaler.transform([game_state])
    state_tensor = torch.FloatTensor(normalized_state)
    
    with torch.no_grad():
        outputs = model(state_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_position = torch.max(outputs, 1)
    
    return predicted_position.item(), probabilities[0].numpy()

# Example usage:
if __name__ == "__main__":
    # Example of how to use this framework
    
    # 1. Prepare your data (you'll replace this with your actual data loading)
    # This is just dummy data to show the structure
    dummy_states = []
    dummy_actions = []
    
    for _ in range(1000):  # Replace with your actual data
        # Example game state
        player_hp = np.random.randint(1, 100)
        enemy_hp = np.random.randint(1, 100)
        hand = [np.random.randint(0, 75) for _ in range(5)]
        effects = [np.random.randint(0, 10) for _ in range(2)]
        
        feature_vector = create_feature_vector(player_hp, enemy_hp, hand, effects)
        dummy_states.append(feature_vector)
        dummy_actions.append(np.random.randint(0, 5))  # Random hand position
    
    # 2. Prepare data
    states, actions, scaler = prepare_data(dummy_states, dummy_actions)
    
    # 3. Create dataset and data loaders
    dataset = CardGameDataset(states, actions)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 4. Create and train model
    input_size = len(dummy_states[0])  # Size of your feature vector
    model = CardGameNet(input_size)
    
    print(f"Model created with input size: {input_size}")
    print(f"Training on {len(train_dataset)} examples")
    print("Starting training...")
    
    train_model(model, train_loader, val_loader, num_epochs=20)
    
    # 5. Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_size': input_size
    }, 'card_game_model.pth')
    
    print("Model saved as 'card_game_model.pth'")
    
    # 6. Example prediction
    example_state = dummy_states[0]
    predicted_pos, probs = make_prediction(model, example_state, scaler)
    print(f"\nExample prediction:")
    print(f"Predicted hand position: {predicted_pos}")
    print(f"Confidence scores: {probs}")
