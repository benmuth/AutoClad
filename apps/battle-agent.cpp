#include <iostream>
#include <string>
#include <cctype>

#include "BattleContext2.h"
#include "GameContext2.h"
#include "SimpleAgent2.h"
#include "AutoClad.h"
#include "../include/utils/scenarios.h"
#include "../include/constants/MonsterEncounters.h"

using namespace sts;

enum class Agent {
  simple,
  autoclad,
};

std::string getAgentName(Agent a) {
    switch (a) {
        case Agent::simple:
            return "SimpleAgent";
        case Agent::autoclad:
            return "AutoClad";
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

void runAgentOnScenario(Agent a, const GameContext& gc, bool printDetails = false, bool generateSnapshot = false, const std::string& snapshotDir = "") {
    std::cout << "Running agent on scenario " << static_cast<int>(gc.info.encounter) << " with seed: " << gc.seed << std::endl;

    // Initialize battle context with the scenario's encounter
    BattleContext initialBc;
    initialBc.init(gc, gc.info.encounter, false);

    // Copy for snapshot (capture initial state)
    BattleContext finalBc = initialBc;

    // sts::search::AutoClad autoclad;
    sts::search::SimpleAgent simple_agent;

    // search::SimpleAgent& agent = (a==Agent::simple ? simple_agent : autoclad);
    search::SimpleAgent& agent = simple_agent;

    // Create and configure the agent
    // sts::search::SimpleAgent agent;

    agent.print = false;

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

    // Run the agent battle simulation
    agent.playoutBattle(finalBc, &snapshot);

    // Get action sequence after battle
    // auto actionSequence = agent.getActionSequence();

    // Report results
    // std::cout << "  Battle Result: ";
    // switch (finalBc.outcome) {
    //     case Outcome::PLAYER_VICTORY:
    //         std::cout << "VICTORY";
    //         break;
    //     case Outcome::PLAYER_LOSS:
    //         std::cout << "DEFEAT";
    //         break;
    //     default:
    //         std::cout << "UNDECIDED";
    //         break;
    // }
    // std::cout << " (Final HP: " << finalBc.player.curHp << "/" << finalBc.player.maxHp
    //               << ", Turns: " << finalBc.turn << ")" << std::endl;

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
    bool generateSnapshots = false;
    std::string snapshotDir = "data/agent_battles";
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
    }
    std::cout << "========================================" << std::endl;

    // Run SimpleAgent on each scenario
    for (const auto& gc : scenarios) {
        runAgentOnScenario(Agent::simple, gc, true, generateSnapshots, snapshotDir);
    }

    std::cout << "All scenarios completed!" << std::endl;
    return 0;
}
