#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "../battle/GameContext2.h"
#include "../battle/BattleContext2.h"
#include "../constants/MonsterEncounters.h"
#include "../constants/Cards.h"
#include "../constants/Relics.h"
#include "../json/single_include/nlohmann/json.hpp"

namespace sts {
namespace utils {

// Helper function to convert card name string to CardId
inline CardId getCardIdFromName(const std::string& cardName) {
    // Linear search through cardEnumStrings array
    // We use the size of the array by calculating how many strings there are
    constexpr int numCards = sizeof(cardEnumStrings) / sizeof(cardEnumStrings[0]);
    for (int i = 0; i < numCards; ++i) {
        if (cardName == cardEnumStrings[i]) {
            return static_cast<CardId>(i);
        }
    }
    return CardId::INVALID;
}

// Helper function to convert relic name string to RelicId
inline RelicId getRelicIdFromName(const std::string& relicName) {
    // Linear search through relicEnumNames array
    constexpr int numRelics = sizeof(relicEnumNames) / sizeof(relicEnumNames[0]);
    for (int i = 0; i < numRelics; ++i) {
        if (relicName == relicEnumNames[i]) {
            return static_cast<RelicId>(i);
        }
    }
    return RelicId::INVALID;
}

// Helper function to format relics with counters for snapshots
inline std::string formatRelicsForSnapshot(const Player& player) {
    std::stringstream relics;
    bool first = true;
    for (int i = 0; i < static_cast<int>(RelicId::INVALID); ++i) {
        RelicId relicId = static_cast<RelicId>(i);
        if (player.hasRelicRuntime(relicId)) {
            if (!first) relics << ", ";
            first = false;
            relics << getRelicName(relicId);

            // Add counter if it exists and is > 0
            int counter = 0;
            switch (relicId) {
                case RelicId::HAPPY_FLOWER: counter = player.happyFlowerCounter; break;
                case RelicId::INCENSE_BURNER: counter = player.incenseBurnerCounter; break;
                case RelicId::INK_BOTTLE: counter = player.inkBottleCounter; break;
                case RelicId::INSERTER: counter = player.inserterCounter; break;
                case RelicId::NUNCHAKU: counter = player.nunchakuCounter; break;
                case RelicId::PEN_NIB: counter = player.penNibCounter; break;
                case RelicId::SUNDIAL: counter = player.sundialCounter; break;
                default: counter = 0; break;
            }
            if (counter > 0) {
                relics << "(" << counter << ")";
            }
        }
    }
    return relics.str();
}

// Helper function to process relics and relic counters from scenario JSON
inline void processRelicsFromScenario(GameContext& gc, const nlohmann::json& scenario) {
    // Process relics if they exist in the scenario
    if (scenario["initial_state"].contains("relics")) {
        for (const auto& relicStr : scenario["initial_state"]["relics"]) {
            std::string relicName = relicStr;
            RelicId relicId = getRelicIdFromName(relicName);
            if (relicId != RelicId::INVALID) {
                RelicInstance relic{relicId, 0};
                gc.relics.add(relic);
            }
        }
    }

    // Process relic counters if they exist in the scenario
    if (scenario["initial_state"].contains("relic_counters")) {
        for (const auto& [relicName, counter] : scenario["initial_state"]["relic_counters"].items()) {
            RelicId relicId = getRelicIdFromName(relicName);
            if (relicId != RelicId::INVALID && gc.relics.has(relicId)) {
                // Find and update the counter for this relic
                for (auto& relic : gc.relics.relics) {
                    if (relic.id == relicId) {
                        relic.data = counter.get<int>();
                        break;
                    }
                }
            }
        }
    }
}

inline GameContext createGameContextFromScenario(const nlohmann::json& scenario) {
    auto seed = scenario["seed"].get<std::uint64_t>();
    auto ascension = scenario["ascension"].get<int>();

    GameContext gc(CharacterClass::IRONCLAD, seed, ascension);

    // Set player stats
    gc.curHp = scenario["initial_state"]["player_hp"];
    gc.maxHp = scenario["initial_state"]["player_max_hp"];

    // Set encounter
    std::string encounterStr = scenario["initial_state"]["encounter"];

    int index = 0;
    for (const char* name : monsterEncounterEnumNames) {
        if (name == encounterStr) {
            break;
        }
        ++index;
      }

    if (index >= 0 && index < sizeof(monsterEncounterEnumNames)/sizeof(monsterEncounterEnumNames[0])) {
         gc.info.encounter = MonsterEncounter(index);
    }


    // Clear deck and add cards from scenario
    gc.deck.cards.clear();
    for (const auto& cardStr : scenario["initial_state"]["deck"]) {
        std::string cardName = cardStr;
        bool isUpgraded = false;

        // Check if card is upgraded (ends with +)
        if (!cardName.empty() && cardName.back() == '+') {
            isUpgraded = true;
            cardName.pop_back(); // Remove the +
        }

        // Handle legacy names for consistency
        if (cardName == "STRIKE") {
            cardName = "STRIKE_RED";
        } else if (cardName == "DEFEND") {
            cardName = "DEFEND_RED";
        }

        CardId cardId = getCardIdFromName(cardName);
        if (cardId != CardId::INVALID) {
            Card card(cardId);
            if (isUpgraded) {
                card.upgrade();
            }
            gc.deck.obtainRaw(card);
        }
    }

    // Process relics and relic counters
    processRelicsFromScenario(gc, scenario);

    return gc;
}

// Overload that takes a seed parameter instead of reading from JSON
inline GameContext createGameContextFromScenario(const nlohmann::json& scenario, std::uint64_t seed) {
    auto ascension = scenario["ascension"].get<int>();

    GameContext gc(CharacterClass::IRONCLAD, seed, ascension);

    gc.floorNum = scenario["floor"];

    // Set player stats
    gc.curHp = scenario["initial_state"]["player_hp"];
    gc.maxHp = scenario["initial_state"]["player_max_hp"];

    // Set encounter
    std::string encounterStr = scenario["initial_state"]["encounter"];

    int index = 0;
    for (const char* name : monsterEncounterEnumNames) {
        if (name == encounterStr) {
            break;
        }
        ++index;
      }

    if (index >= 0 && index < sizeof(monsterEncounterEnumNames)/sizeof(monsterEncounterEnumNames[0])) {
         gc.info.encounter = MonsterEncounter(index);
    }


    // Clear deck and add cards from scenario
    gc.deck.cards.clear();
    for (const auto& cardStr : scenario["initial_state"]["deck"]) {
        std::string cardName = cardStr;
        bool isUpgraded = false;

        // Check if card is upgraded (ends with +)
        if (!cardName.empty() && cardName.back() == '+') {
            isUpgraded = true;
            cardName.pop_back(); // Remove the +
        }

        // Handle legacy names for consistency
        if (cardName == "STRIKE") {
            cardName = "STRIKE_RED";
        } else if (cardName == "DEFEND") {
            cardName = "DEFEND_RED";
        }

        CardId cardId = getCardIdFromName(cardName);
        if (cardId != CardId::INVALID) {
            Card card(cardId);
            if (isUpgraded) {
                card.upgrade();
            }
            gc.deck.obtainRaw(card);
        }
    }

    // Process relics and relic counters
    processRelicsFromScenario(gc, scenario);

    return gc;
}

inline std::vector<GameContext> loadScenariosFromDirectory(const std::string& directoryPath) {
    std::vector<GameContext> gameContexts;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.path().extension() == ".json") {
                try {
                    std::ifstream file(entry.path());
                    if (file.is_open()) {
                        nlohmann::json scenario;
                        file >> scenario;

                        // Generate a seed based on scenario name hash or use a default
                        std::uint64_t seed = 12345;
                        if (scenario.contains("name")) {
                            std::string name = scenario["name"];
                            seed = std::hash<std::string>{}(name);
                        }
                        GameContext gc = createGameContextFromScenario(scenario, seed);
                        gameContexts.push_back(gc);
                    }
                } catch (const std::exception& e) {
                    // Log the error for this specific file and continue
                    std::cerr << "Error loading scenario file " << entry.path() << ": " << e.what() << std::endl;
                    continue;
                }
            }
        }
    } catch (const std::exception& e) {
        // Handle filesystem errors
        std::cerr << "Error accessing directory " << directoryPath << ": " << e.what() << std::endl;
    }

    return gameContexts;
}

// Snapshot formatting helper functions

inline bool isAttackCard(CardId id) {
    return id == CardId::STRIKE_RED || id == CardId::BASH ||
           id == CardId::STRIKE_GREEN || id == CardId::STRIKE_BLUE || id == CardId::STRIKE_PURPLE;
}

inline bool isDefendCard(CardId id) {
    return id == CardId::DEFEND_RED || id == CardId::DEFEND_GREEN ||
           id == CardId::DEFEND_BLUE || id == CardId::DEFEND_PURPLE;
}

inline std::string formatBattleSnapshot(const GameContext& gc, const BattleContext& initialBc, const BattleContext& finalBc, const std::string& scenarioName = "Agent Battle", const std::string& agentName = "Unknown") {
    std::stringstream snapshot;

    // Header
    snapshot << "=== COMBAT: \"" << scenarioName << "\" ===" << std::endl;
    snapshot << "Agent: " << agentName << std::endl;
    snapshot << "Seed: " << gc.seed << " | Ascension: " << gc.ascension 
             << " | Floor: " << gc.floorNum << std::endl << std::endl;

    // Initial State
    snapshot << "Initial State:" << std::endl;
    snapshot << "  Player: " << initialBc.player.curHp << "/" << initialBc.player.maxHp 
             << " HP, " << initialBc.player.energy << " Energy" << std::endl;

    // Format monster
    if (initialBc.monsters.monsterCount > 0) {
        const auto& monster = initialBc.monsters.arr[0];
        snapshot << "  Enemy: " << monsterEncounterStrings[static_cast<int>(gc.info.encounter)]
                 << " (" << monster.curHp << " HP)";
        snapshot << std::endl;
    }

    // Format hand
    snapshot << "  Hand: ";
    for (int i = 0; i < initialBc.cards.cardsInHand; ++i) {
        if (i > 0) snapshot << ", ";
        const auto& card = initialBc.cards.hand[i];
        snapshot << getCardName(card.getId()) << "(" << int(card.cost) << ")";
    }
    snapshot << std::endl;

    snapshot << "  Deck: " << initialBc.cards.drawPile.size() << " cards remaining" << std::endl;

    // Format relics
    snapshot << "  Relics: " << formatRelicsForSnapshot(initialBc.player) << std::endl;

    snapshot << std::endl;

    // Final Result
    snapshot << "Final Result:" << std::endl;
    snapshot << "  Outcome: ";
    switch (finalBc.outcome) {
        case Outcome::PLAYER_VICTORY:
            snapshot << "PLAYER_VICTORY" << std::endl;
            break;
        case Outcome::PLAYER_LOSS:
            snapshot << "PLAYER_LOSS" << std::endl;
            break;
        default:
            snapshot << "UNDECIDED" << std::endl;
            break;
    }

    snapshot << "  Player HP: " << finalBc.player.curHp << "/" << finalBc.player.maxHp << std::endl;
    snapshot << "  Relics: " << formatRelicsForSnapshot(finalBc.player) << std::endl;
    snapshot << "  Turns: " << finalBc.turn << std::endl;

    // RNG counters for determinism verification
    snapshot << "  RNG Counters: shuffle=" << finalBc.shuffleRng.counter 
             << ", cardRandom=" << finalBc.cardRandomRng.counter 
             << ", misc=" << finalBc.miscRng.counter << std::endl;

    return snapshot.str();
}

inline void writeSnapshotToFile(const std::string& snapshot, const std::string& filePath) {
    // Create directory if it doesn't exist
    std::filesystem::path path(filePath);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream outFile(filePath);
    outFile << snapshot;
    outFile.close();
}

} // namespace utils
} // namespace sts
