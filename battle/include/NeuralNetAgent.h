//
// Created by Claude on 2025-01-09.
//

#ifndef STS_LIGHTSPEED_NEURALNETAGENT_H
#define STS_LIGHTSPEED_NEURALNETAGENT_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <memory>

#include "GameContext2.h"
#include "Action2.h"

// Forward declarations to avoid conflicts
namespace sts {
    class BattleContext;
}

namespace sts::search {

    struct NeuralNetAgent {
        
        std::vector<int> actionHistory;
        GameContext *curGameContext; // unsafe only use in private methods during playout
        
        fixed_list<int,16> mapPath;
        
        bool print = false;
        
        // Neural network components
        torch::jit::script::Module model;
        std::vector<float> scaler_mean;
        std::vector<float> scaler_scale;
        int input_size;
        bool model_loaded = false;
        
        NeuralNetAgent();
        ~NeuralNetAgent() = default;
        
        // Load the neural network model and scaler parameters
        bool loadModel(const std::string& model_path, const std::string& scaler_path);
        
        // Extract features from battle context (matching Python parser)
        std::vector<float> extractFeatures(const BattleContext &bc) const;
        
        // Normalize features using scaler parameters
        void normalizeFeatures(std::vector<float>& features) const;
        
        // Make prediction using neural network
        int predictAction(const BattleContext &bc);
        
        // Get action probabilities
        std::vector<float> getActionProbabilities(const BattleContext &bc);
        
        // Validate if predicted action is valid
        bool isValidAction(const BattleContext &bc, int action) const;
        
        // Fallback to SimpleAgent logic if needed
        int getFallbackAction(const BattleContext &bc);
        
        [[nodiscard]] int getIncomingDamage(const BattleContext &bc) const;
        
        void takeAction(BattleContext &bc, Action a);
        void playoutBattle(BattleContext &bc, std::stringstream* snapshot = nullptr);
        
        // Main decision function - chooses card to play using neural network
        virtual void stepBattleCardPlay(BattleContext &bc);
        void stepBattleCardSelect(BattleContext &bc);
        
        bool playPotion(BattleContext &bc);
        static void runAgentsMt(int threadCount, std::uint64_t startSeed, int playoutCount, bool print);
        
        // Get action sequence as a simple list of action descriptions
        std::vector<std::string> getActionSequence() const;
    };

}

#endif //STS_LIGHTSPEED_NEURALNETAGENT_H