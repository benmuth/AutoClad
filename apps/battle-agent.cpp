#include <iostream>
#include <string>
#include <cctype>
#include <filesystem>

#include "BattleContext2.h"
#include "GameContext2.h"
#include "SimpleAgent2.h"
#include "AutoClad.h"
#include "NeuralNetAgent.h"
#include "../include/utils/scenarios.h"
#include "../include/constants/MonsterEncounters.h"

using namespace sts;

enum class Agent {
  simple,
  autoclad,
  neural,
};

std::string getAgentName(Agent a) {
    switch (a) {
        case Agent::simple:
            return "SimpleAgent";
        case Agent::autoclad:
            return "AutoClad";
        case Agent::neural:
            return "NeuralNetAgent";
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
    sts::search::NeuralNetAgent neural_agent;

    // Configure agents
    simple_agent.print = false;
    autoclad.print = false;
    neural_agent.print = false;

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
        case Agent::neural:
            neural_agent.playoutBattle(finalBc, &snapshot);
            break;
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

Agent parseAgent(const std::string& agentStr) {
    std::string lowerAgent = agentStr;
    std::transform(lowerAgent.begin(), lowerAgent.end(), lowerAgent.begin(), ::tolower);
    
    if (lowerAgent == "simple") {
        return Agent::simple;
    } else if (lowerAgent == "autoclad") {
        return Agent::autoclad;
    } else if (lowerAgent == "neural") {
        return Agent::neural;
    } else {
        throw std::invalid_argument("Invalid agent type: " + agentStr + ". Valid options are: simple, autoclad, neural");
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " --agent=<agent_type> [options]" << std::endl;
    std::cout << "Required arguments:" << std::endl;
    std::cout << "  --agent=<type>      Agent type (simple, autoclad, neural)" << std::endl;
    std::cout << "Optional arguments:" << std::endl;
    std::cout << "  --snapshot[=<dir>]  Generate snapshots (default: data/agent_battles/<agent>)" << std::endl;
    std::cout << "  --scenario=<name>   Filter scenarios by name (can be used multiple times)" << std::endl;
}

int main(int argc, char* argv[]) {
    bool generateSnapshots = true;
    std::string snapshotDir = "";
    std::vector<std::string> scenarioFilters;
    Agent selectedAgent;
    bool agentSpecified = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.length() > 8 && arg.substr(0, 8) == "--agent=") {
            std::string agentValue = arg.substr(8); // Remove "--agent="
            try {
                selectedAgent = parseAgent(agentValue);
                agentSpecified = true;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "--snapshot") {
            generateSnapshots = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                snapshotDir = argv[i + 1];
                ++i; // Skip the directory argument
            }
        } else if (arg.length() > 11 && arg.substr(0, 11) == "--snapshot=") {
            generateSnapshots = true;
            snapshotDir = arg.substr(11); // Remove "--snapshot="
        } else if (arg.length() > 11 && arg.substr(0, 11) == "--scenario=") {
            std::string scenarioValue = arg.substr(11); // Remove "--scenario="
            scenarioFilters.push_back(scenarioValue);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Check if agent was specified
    if (!agentSpecified) {
        std::cerr << "Error: Agent type must be specified with --agent=<type>" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Set default snapshot directory based on agent if not specified
    if (snapshotDir.empty()) {
        std::string agentName = getAgentName(selectedAgent);
        std::transform(agentName.begin(), agentName.end(), agentName.begin(), ::tolower);
        snapshotDir = "data/agent_battles/" + agentName;
    }

    // Load all scenarios from multiple directories
    // std::vector<GameContext> allScenarios = sts::utils::loadScenariosFromDirectory("battle/scenarios/");
    std::vector<std::string> scenarioDirs = {
        "battle/generated_scenarios/jaw_worm/",
        "battle/randomized_scenarios/"
    };
    std::vector<GameContext> allScenarios = sts::utils::loadScenariosFromMultipleDirectories(scenarioDirs);


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

    std::cout << "Using " << getAgentName(selectedAgent) << " for battle simulations" << std::endl;
    
    for (const auto& gc : scenarios) {
        runAgentOnScenario(selectedAgent, gc, true, generateSnapshots, snapshotDir);
    }

    std::cout << "All scenarios completed!" << std::endl;
    return 0;
}
