#include <iostream>
#include <string>
#include <cctype>

#include "BattleContext2.h"
#include "GameContext2.h"
#include "SimpleAgent2.h"
#include "AutoClad.h"
#ifdef NEURAL_NET_ENABLED
#include "NeuralNetAgent.h"
#endif
#include "../include/utils/scenarios.h"
#include "../include/constants/MonsterEncounters.h"

using namespace sts;

enum class Agent {
  simple,
  autoclad,
#ifdef NEURAL_NET_ENABLED
  neural,
#endif
};

std::string getAgentName(Agent a) {
    switch (a) {
        case Agent::simple:
            return "SimpleAgent";
        case Agent::autoclad:
            return "AutoClad";
#ifdef NEURAL_NET_ENABLED
        case Agent::neural:
            return "NeuralNetAgent";
#endif
        default:
            return "Unknown";
    }
}

std::vector<GameContext> filterScenarios(const std::vector<GameContext>& allScenarios, const std::vector<std::string>& filters) {
    // If no filters specified or "all" is specified, return all scenarios
    if (filters.empty() || (filters.size() == 1 && filters[0] == "all")) {
        return allScenarios;
    }

    std::vector<GameContext> filteredScenarios;
    for (const auto& gc : allScenarios) {
        // Get the encounter name for comparison
        std::string encounterName = monsterEncounterStrings[static_cast<int>(gc.info.encounter)];

        // Convert encounter name to lowercase for case-insensitive matching
        std::string lowerEncounterName = encounterName;
        std::transform(lowerEncounterName.begin(), lowerEncounterName.end(), lowerEncounterName.begin(), ::tolower);

        // Check if this scenario matches any of the filters
        for (const auto& filter : filters) {
            std::string lowerFilter = filter;
            std::transform(lowerFilter.begin(), lowerFilter.end(), lowerFilter.begin(), ::tolower);

            // Match by encounter name (with spaces replaced by underscores for command line friendliness)
            std::string underscoreEncounterName = lowerEncounterName;
            std::replace(underscoreEncounterName.begin(), underscoreEncounterName.end(), ' ', '_');

            if (lowerFilter == underscoreEncounterName || lowerFilter == lowerEncounterName) {
                filteredScenarios.push_back(gc);
                break; // Don't add the same scenario multiple times
            }
        }
    }

    return filteredScenarios;
}

void clearSnapshotDirectory(const std::string& snapshotDir) {
    try {
        if (std::filesystem::exists(snapshotDir)) {
            // Remove all files in the directory
            for (const auto& entry : std::filesystem::directory_iterator(snapshotDir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".snap") {
                    std::filesystem::remove(entry.path());
                }
            }
            std::cout << "Cleared existing snapshots from: " << snapshotDir << std::endl;
        } else {
            // Create the directory if it doesn't exist
            std::filesystem::create_directories(snapshotDir);
            std::cout << "Created snapshot directory: " << snapshotDir << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Error managing snapshot directory: " << ex.what() << std::endl;
    }
}

void runAgentOnScenario(Agent a, const GameContext& gc, bool printDetails = false, bool generateSnapshot = false, const std::string& snapshotDir = "") {
    std::cout << "Running agent on scenario " << static_cast<int>(gc.info.encounter) << " with seed: " << gc.seed << std::endl;

    // Initialize battle context with the scenario's encounter
    BattleContext initialBc;
    initialBc.init(gc, gc.info.encounter, false);

    // Copy for snapshot (capture initial state)
    BattleContext finalBc = initialBc;

    // Initialize agents based on selected type
    sts::search::SimpleAgent simple_agent;
    sts::search::AutoClad autoclad;
#ifdef NEURAL_NET_ENABLED
    sts::search::NeuralNetAgent neural_agent;
#endif

    // Configure agents
    simple_agent.print = false;
    autoclad.print = false;
#ifdef NEURAL_NET_ENABLED
    neural_agent.print = false;
#endif

    if (printDetails) {
        std::cout << "  AGENT: " << getAgentName(a) << std::endl;
        std::cout << "  Initial State:" << std::endl;
        std::cout << "    Encounter: " << static_cast<int>(gc.info.encounter) << std::endl;
        std::cout << "    Player HP: " << initialBc.player.curHp << "/" << initialBc.player.maxHp << std::endl;
        std::cout << "    Hand size: " << initialBc.cards.cardsInHand << std::endl;
        std::cout << "    Deck size: " << initialBc.cards.drawPile.size() << std::endl;
        std::cout << "  Starting battle..." << std::endl;
    }

    std::stringstream snapshot;

    // Run the agent battle simulation based on selected type
    switch (a) {
        case Agent::simple:
            simple_agent.playoutBattle(finalBc, &snapshot);
            break;
        case Agent::autoclad:
            autoclad.playoutBattle(finalBc, &snapshot);
            break;
#ifdef NEURAL_NET_ENABLED
        case Agent::neural:
            neural_agent.playoutBattle(finalBc, &snapshot);
            break;
#endif
        default:
            simple_agent.playoutBattle(finalBc, &snapshot);
            break;
    }

    // Get action sequence after battle
    // auto actionSequence = agent.getActionSequence();

    // Report results
    if (printDetails) {
        std::cout << "  Battle Result: ";
        switch (finalBc.outcome) {
            case Outcome::PLAYER_VICTORY:
                std::cout << "VICTORY";
                break;
            case Outcome::PLAYER_LOSS:
                std::cout << "DEFEAT";
                break;
            default:
                std::cout << "UNDECIDED";
                break;
        }
        std::cout << " (Final HP: " << finalBc.player.curHp << "/" << finalBc.player.maxHp
                  << ", Turns: " << finalBc.turn << ")" << std::endl;
    }

    // Print action sequence
    // if (!actionSequence.empty()) {
    //     std::cout << "  Action Sequence (" << actionSequence.size() << " actions): ";
    //     for (size_t i = 0; i < actionSequence.size(); ++i) {
    //         if (i > 0) std::cout << " -> ";
    //         std::cout << actionSequence[i];
    //     }
    //     std::cout << std::endl;
    // }

    // Generate snapshot if requested
    if (generateSnapshot && !snapshotDir.empty()) {
        std::string agentName = getAgentName(a);
        std::string encounterName = monsterEncounterStrings[static_cast<int>(gc.info.encounter)];
        std::string scenarioName = agentName + "_vs_" + encounterName + "_" + std::to_string(gc.seed);
        std::string snapshot_str = utils::formatBattleSnapshot(gc, initialBc, finalBc, &snapshot, scenarioName, agentName);

        std::string filename = scenarioName + ".snap";
        std::string filepath = snapshotDir + "/" + filename;

        utils::writeSnapshotToFile(snapshot_str, filepath);
        std::cout << "  Snapshot written to: " << filepath << std::endl;
    }

    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    bool generateSnapshots = true; // Always generate snapshots for neural network analysis
    std::string snapshotDir = "data/agent_battles/nn";
    std::vector<std::string> scenarioFilters;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--snapshot") {
            generateSnapshots = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                snapshotDir = argv[i + 1];
                ++i; // Skip the directory argument
            }
        } else if (arg.length() > 11 && arg.substr(0, 11) == "--scenario=") {
            std::string scenarioValue = arg.substr(11); // Remove "--scenario="
            scenarioFilters.push_back(scenarioValue);
        }
    }

    // Load all scenarios from the scenarios directory
    // std::vector<GameContext> allScenarios = sts::utils::loadScenariosFromDirectory("battle/scenarios/");
    std::vector<GameContext> allScenarios = sts::utils::loadScenariosFromDirectory("battle/generated_scenarios/jaw_worm/");


    // Filter scenarios based on command line arguments
    std::vector<GameContext> scenarios = filterScenarios(allScenarios, scenarioFilters);

    std::cout << "Loaded " << allScenarios.size() << " total scenarios";
    if (!scenarioFilters.empty()) {
        std::cout << ", filtered to " << scenarios.size() << " scenarios";
        std::cout << " (filters: ";
        for (size_t i = 0; i < scenarioFilters.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << scenarioFilters[i];
        }
        std::cout << ")";
    }
    std::cout << std::endl;

    if (generateSnapshots) {
        std::cout << "Snapshots will be written to: " << snapshotDir << std::endl;
        clearSnapshotDirectory(snapshotDir);
    }
    std::cout << "========================================" << std::endl;

    // Run neural agent on each scenario (fallback to simple agent if neural not available)
    Agent selectedAgent = Agent::simple;
#ifdef NEURAL_NET_ENABLED
    selectedAgent = Agent::neural;
    std::cout << "Using NeuralNetAgent for battle simulations" << std::endl;
#else
    std::cout << "Using SimpleAgent for battle simulations (neural network not available)" << std::endl;
#endif
    
    for (const auto& gc : scenarios) {
        runAgentOnScenario(selectedAgent, gc, true, generateSnapshots, snapshotDir);
    }

    std::cout << "All scenarios completed!" << std::endl;
    return 0;
}
