//
// Created by Claude on 2025-01-09.
//

#include "NeuralNetAgent.h"
#include "SimpleAgent2.h"
#include "Game2.h"
#include "BattleContext2.h"
#include "GameContext2.h"
#include "BattleSimulator2.h"
#include "CardInstance2.h"
#include "../../include/constants/CharacterClasses.h"
#include "../../include/constants/MonsterEncounters.h"
#include "../../include/game/Card.h"
#include "../../include/sim/PrintHelpers.h"
#include "Action2.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace sts;
using json = nlohmann::json;

namespace sts::search {

NeuralNetAgent::NeuralNetAgent() 
    : actionHistory(), curGameContext(nullptr), mapPath(), print(false),
      input_size(15), model_loaded(false) {
    
    // Try to load model at construction time
    loadModel("jaw_worm_model_traced.pt", "scaler_params.json");
}

bool NeuralNetAgent::loadModel(const std::string& model_path, const std::string& scaler_path) {
    try {
        // Load the TorchScript model using LibTorch C++ API
        model = torch::jit::load(model_path);
        model.eval();
        
        // Optimize for inference
        model = torch::jit::optimize_for_inference(model);
        
        // Load scaler parameters
        std::ifstream scaler_file(scaler_path);
        if (!scaler_file.is_open()) {
            std::cerr << "Could not open scaler parameters file: " << scaler_path << std::endl;
            return false;
        }
        
        json scaler_json;
        scaler_file >> scaler_json;
        
        scaler_mean = scaler_json["mean"].get<std::vector<float>>();
        scaler_scale = scaler_json["scale"].get<std::vector<float>>();
        input_size = scaler_json["input_size"].get<int>();
        
        // Validate dimensions
        if (scaler_mean.size() != input_size || scaler_scale.size() != input_size) {
            std::cerr << "Scaler dimensions mismatch: expected " << input_size 
                      << " but got mean=" << scaler_mean.size() 
                      << " scale=" << scaler_scale.size() << std::endl;
            return false;
        }
        
        model_loaded = true;
        
        if (print) {
            std::cout << "LibTorch neural network model loaded successfully" << std::endl;
            std::cout << "Input size: " << input_size << std::endl;
            std::cout << "Model optimized for inference" << std::endl;
        }
        
        return true;
        
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error loading model: " << e.msg() << std::endl;
        model_loaded = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading neural network model: " << e.what() << std::endl;
        model_loaded = false;
        return false;
    }
}

std::vector<float> NeuralNetAgent::extractFeatures(const BattleContext &bc) const {
    std::vector<float> features;
    features.reserve(15); // We know we need exactly 15 features
    
    // Feature 0: turn
    features.push_back(static_cast<float>(bc.turn));
    
    // Feature 1: health
    features.push_back(static_cast<float>(bc.player.curHp));
    
    // Feature 2: maxhealth
    features.push_back(static_cast<float>(bc.player.maxHp));
    
    // Feature 3: energy
    features.push_back(static_cast<float>(bc.player.energy));
    
    // Feature 4: block
    features.push_back(static_cast<float>(bc.player.block));
    
    // Feature 5: enemy0_hp (assuming single enemy for Jaw Worm)
    if (bc.monsters.monstersAlive > 0) {
        features.push_back(static_cast<float>(bc.monsters.arr[0].curHp));
    } else {
        features.push_back(0.0f);
    }
    
    // Features 6-10: hand_card0 to hand_card4 (5 hand slots)
    for (int i = 0; i < 5; ++i) {
        if (i < bc.cards.cardsInHand) {
            // Use card ID as feature (matching Python parser behavior)
            features.push_back(static_cast<float>(static_cast<int>(bc.cards.hand[i].getId())));
        } else {
            features.push_back(-1.0f); // Empty slot
        }
    }
    
    // Feature 11: draw_size
    features.push_back(static_cast<float>(bc.cards.drawPile.size()));
    
    // Feature 12: discard_size
    features.push_back(static_cast<float>(bc.cards.discardPile.size()));
    
    // Feature 13: exhaust_size
    features.push_back(static_cast<float>(bc.cards.exhaustPile.size()));
    
    // Feature 14: potion_count (count non-empty potions)
    int potion_count = 0;
    for (int i = 0; i < bc.potionCapacity; ++i) {
        if (bc.potions[i] != Potion::EMPTY_POTION_SLOT) {
            potion_count++;
        }
    }
    features.push_back(static_cast<float>(potion_count));
    
    return features;
}

void NeuralNetAgent::normalizeFeatures(std::vector<float>& features) const {
    if (features.size() != scaler_mean.size() || features.size() != scaler_scale.size()) {
        throw std::runtime_error("Feature size mismatch with scaler parameters");
    }
    
    for (size_t i = 0; i < features.size(); ++i) {
        features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

int NeuralNetAgent::predictAction(const BattleContext &bc) {
    if (!model_loaded) {
        return getFallbackAction(bc);
    }
    
    try {
        // Extract and normalize features
        std::vector<float> features = extractFeatures(bc);
        normalizeFeatures(features);
        
        // Create tensor using LibTorch C++ API - no data copy, direct memory mapping
        torch::Tensor input_tensor = torch::from_blob(
            features.data(), 
            {1, static_cast<long>(input_size)}, 
            torch::TensorOptions().dtype(torch::kFloat32)
        );
        
        // Run inference with no_grad for performance
        torch::NoGradGuard no_grad;
        torch::Tensor output = model.forward({input_tensor}).toTensor();
        
        // Get predicted action (argmax) - direct CPU operation
        int predicted_action = output.argmax(-1).item<int>();
        
        // Validate action and use fallback if needed
        if (isValidAction(bc, predicted_action)) {
            return predicted_action;
        } else {
            if (print) {
                std::cout << "Invalid action " << predicted_action << " predicted, using fallback" << std::endl;
            }
            return getFallbackAction(bc);
        }
        
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error in prediction: " << e.msg() << std::endl;
        return getFallbackAction(bc);
    } catch (const std::exception& e) {
        std::cerr << "Error in neural network prediction: " << e.what() << std::endl;
        return getFallbackAction(bc);
    }
}

std::vector<float> NeuralNetAgent::getActionProbabilities(const BattleContext &bc) {
    std::vector<float> probs(6, 0.0f); // 6 actions: 5 hand positions + end turn
    
    if (!model_loaded) {
        // Return uniform probabilities as fallback
        std::fill(probs.begin(), probs.end(), 1.0f / 6.0f);
        return probs;
    }
    
    try {
        // Extract and normalize features
        std::vector<float> features = extractFeatures(bc);
        normalizeFeatures(features);
        
        // Create tensor using LibTorch C++ API
        torch::Tensor input_tensor = torch::from_blob(
            features.data(), 
            {1, static_cast<long>(input_size)}, 
            torch::TensorOptions().dtype(torch::kFloat32)
        );
        
        // Run inference with no_grad for performance
        torch::NoGradGuard no_grad;
        torch::Tensor output = model.forward({input_tensor}).toTensor();
        torch::Tensor probabilities = torch::softmax(output, -1);
        
        // Extract probabilities efficiently
        auto prob_accessor = probabilities.accessor<float, 2>();
        for (int i = 0; i < 6; ++i) {
            probs[i] = prob_accessor[0][i];
        }
        
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error getting probabilities: " << e.msg() << std::endl;
        std::fill(probs.begin(), probs.end(), 1.0f / 6.0f);
    } catch (const std::exception& e) {
        std::cerr << "Error getting action probabilities: " << e.what() << std::endl;
        std::fill(probs.begin(), probs.end(), 1.0f / 6.0f);
    }
    
    return probs;
}

bool NeuralNetAgent::isValidAction(const BattleContext &bc, int action) const {
    // Actions 0-4: play hand cards
    if (action >= 0 && action <= 4) {
        return (action < bc.cards.cardsInHand && bc.cards.hand[action].canUseOnAnyTarget(bc));
    }
    
    // Action 5: end turn (always valid)
    if (action == 5) {
        return true;
    }
    
    return false;
}

int NeuralNetAgent::getFallbackAction(const BattleContext &bc) {
    // Simple fallback: play first playable card or end turn
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        if (bc.cards.hand[i].canUseOnAnyTarget(bc)) {
            return i;
        }
    }
    return 5; // End turn
}

int NeuralNetAgent::getIncomingDamage(const BattleContext &bc) const {
    // Reuse SimpleAgent's implementation
    int totalIncomingDamage = 0;
    for (int i = 0; i < bc.monsters.monstersAlive; ++i) {
        const auto &monster = bc.monsters.arr[i];
        if (!monster.isDeadOrEscaped()) {
            // This is a simplified version - you might want to copy the full logic from SimpleAgent2
            totalIncomingDamage += 10; // Placeholder - replace with actual damage calculation
        }
    }
    return totalIncomingDamage;
}

void NeuralNetAgent::stepBattleCardPlay(BattleContext &bc) {
    if (bc.player.energy <= 0 || bc.cards.cardsInHand == 0) {
        takeAction(bc, Action(ActionType::END_TURN));
        return;
    }
    
    // Use neural network to predict action
    int predicted_action = predictAction(bc);
    
    if (print) {
        std::vector<float> probs = getActionProbabilities(bc);
        std::cout << "Neural Network Decision:" << std::endl;
        std::cout << "  Predicted action: " << predicted_action;
        if (predicted_action < 5) {
            std::cout << " (Play hand card " << predicted_action << ")";
        } else {
            std::cout << " (End turn)";
        }
        std::cout << std::endl;
        
        std::cout << "  Confidence scores:" << std::endl;
        const std::vector<std::string> action_names = {
            "Hand 0", "Hand 1", "Hand 2", "Hand 3", "Hand 4", "End Turn"
        };
        for (int i = 0; i < 6; ++i) {
            std::cout << "    " << action_names[i] << ": " 
                      << std::fixed << std::setprecision(1) << (probs[i] * 100.0f) << "%";
            if (i == predicted_action) std::cout << " ←";
            std::cout << std::endl;
        }
    }
    
    // Execute the predicted action
    if (predicted_action == 5) {
        // End turn
        takeAction(bc, Action(ActionType::END_TURN));
    } else if (predicted_action >= 0 && predicted_action < bc.cards.cardsInHand) {
        // Play the predicted card
        takeAction(bc, Action(ActionType::CARD, predicted_action));
    } else {
        // Fallback: end turn
        takeAction(bc, Action(ActionType::END_TURN));
    }
}

void NeuralNetAgent::takeAction(BattleContext &bc, Action a) {
    // Use the correct Action2 API
    a.execute(bc);
}

void NeuralNetAgent::playoutBattle(BattleContext &bc, std::stringstream* snapshot) {
    // Copy the playout pattern from SimpleAgent2 exactly, including detailed snapshot recording
    while (bc.outcome == Outcome::UNDECIDED) {
        // Record detailed battle state data in the same format as SimpleAgent2
        if (snapshot && bc.inputState == InputState::PLAYER_NORMAL) {
            *snapshot << bc << std::endl;
            *snapshot << "ActionQueue: " << std::endl;
            if (!bc.lastPlayerActionDescription.empty()) {
                *snapshot << "Last Player Action: " << bc.lastPlayerActionDescription << std::endl << std::endl;
            } else {
                *snapshot << "No last action" << bc.lastPlayerActionDescription << std::endl << std::endl;
            }
        }
        
        if (bc.inputState == InputState::CARD_SELECT) {
            stepBattleCardSelect(bc);
        } else if (bc.inputState == InputState::PLAYER_NORMAL) {
            stepBattleCardPlay(bc);
        }
        
        bc.executeActions();
    }
}

void NeuralNetAgent::stepBattleCardSelect(BattleContext &bc) {
    // Simple implementation - select first card (can be improved)
    if (bc.cardSelectInfo.canPickZero) {
        // Can choose to pick nothing
        takeAction(bc, Action(ActionType::MULTI_CARD_SELECT, 0));
    } else {
        // Must pick at least one card - pick first available
        takeAction(bc, Action(ActionType::SINGLE_CARD_SELECT, 0));
    }
}

bool NeuralNetAgent::playPotion(BattleContext &bc) {
    // Simple implementation - don't use potions for now
    return false;
}

void NeuralNetAgent::runAgentsMt(int threadCount, std::uint64_t startSeed, int playoutCount, bool print) {
    // Multi-threaded agent runner - implement based on SimpleAgent2 pattern
    // For now, provide a basic implementation
    for (int i = 0; i < playoutCount; ++i) {
        // Create battle context and run single playout
        // This would need proper implementation based on the existing pattern
    }
}

std::vector<std::string> NeuralNetAgent::getActionSequence() const {
    std::vector<std::string> sequence;
    for (int action : actionHistory) {
        if (action < 5) {
            sequence.push_back("PLAY_CARD_" + std::to_string(action));
        } else {
            sequence.push_back("END_TURN");
        }
    }
    return sequence;
}

} // namespace sts::search
