#include "FightDefinition.h"
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace battle {

std::unordered_map<std::string, sts::MonsterEncounter> createEncounterMap() {
    return {
        {"INVALID", sts::MonsterEncounter::INVALID},
        {"CULTIST", sts::MonsterEncounter::CULTIST},
        {"JAW_WORM", sts::MonsterEncounter::JAW_WORM},
        {"TWO_LOUSE", sts::MonsterEncounter::TWO_LOUSE},
        {"SMALL_SLIMES", sts::MonsterEncounter::SMALL_SLIMES},
        {"BLUE_SLAVER", sts::MonsterEncounter::BLUE_SLAVER},
        {"GREMLIN_GANG", sts::MonsterEncounter::GREMLIN_GANG},
        {"LOOTER", sts::MonsterEncounter::LOOTER},
        {"LARGE_SLIME", sts::MonsterEncounter::LARGE_SLIME},
        {"LOTS_OF_SLIMES", sts::MonsterEncounter::LOTS_OF_SLIMES},
        {"EXORDIUM_THUGS", sts::MonsterEncounter::EXORDIUM_THUGS},
        {"EXORDIUM_WILDLIFE", sts::MonsterEncounter::EXORDIUM_WILDLIFE},
        {"RED_SLAVER", sts::MonsterEncounter::RED_SLAVER},
        {"THREE_LOUSE", sts::MonsterEncounter::THREE_LOUSE},
        {"TWO_FUNGI_BEASTS", sts::MonsterEncounter::TWO_FUNGI_BEASTS},
        {"GREMLIN_NOB", sts::MonsterEncounter::GREMLIN_NOB},
        {"LAGAVULIN", sts::MonsterEncounter::LAGAVULIN},
        {"THREE_SENTRIES", sts::MonsterEncounter::THREE_SENTRIES},
        {"SLIME_BOSS", sts::MonsterEncounter::SLIME_BOSS},
        {"THE_GUARDIAN", sts::MonsterEncounter::THE_GUARDIAN},
        {"HEXAGHOST", sts::MonsterEncounter::HEXAGHOST}
    };
}

std::unordered_map<sts::MonsterEncounter, std::string> createReverseEncounterMap() {
    return {
        {sts::MonsterEncounter::INVALID, "INVALID"},
        {sts::MonsterEncounter::CULTIST, "CULTIST"},
        {sts::MonsterEncounter::JAW_WORM, "JAW_WORM"},
        {sts::MonsterEncounter::TWO_LOUSE, "TWO_LOUSE"},
        {sts::MonsterEncounter::SMALL_SLIMES, "SMALL_SLIMES"},
        {sts::MonsterEncounter::BLUE_SLAVER, "BLUE_SLAVER"},
        {sts::MonsterEncounter::GREMLIN_GANG, "GREMLIN_GANG"},
        {sts::MonsterEncounter::LOOTER, "LOOTER"},
        {sts::MonsterEncounter::LARGE_SLIME, "LARGE_SLIME"},
        {sts::MonsterEncounter::LOTS_OF_SLIMES, "LOTS_OF_SLIMES"},
        {sts::MonsterEncounter::EXORDIUM_THUGS, "EXORDIUM_THUGS"},
        {sts::MonsterEncounter::EXORDIUM_WILDLIFE, "EXORDIUM_WILDLIFE"},
        {sts::MonsterEncounter::RED_SLAVER, "RED_SLAVER"},
        {sts::MonsterEncounter::THREE_LOUSE, "THREE_LOUSE"},
        {sts::MonsterEncounter::TWO_FUNGI_BEASTS, "TWO_FUNGI_BEASTS"},
        {sts::MonsterEncounter::GREMLIN_NOB, "GREMLIN_NOB"},
        {sts::MonsterEncounter::LAGAVULIN, "LAGAVULIN"},
        {sts::MonsterEncounter::THREE_SENTRIES, "THREE_SENTRIES"},
        {sts::MonsterEncounter::SLIME_BOSS, "SLIME_BOSS"},
        {sts::MonsterEncounter::THE_GUARDIAN, "THE_GUARDIAN"},
        {sts::MonsterEncounter::HEXAGHOST, "HEXAGHOST"}
    };
}

sts::MonsterEncounter parseMonsterEncounter(const std::string& encounter_str) {
    static auto encounterMap = createEncounterMap();
    auto it = encounterMap.find(encounter_str);
    if (it != encounterMap.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unknown monster encounter: " + encounter_str);
}

std::string monsterEncounterToString(sts::MonsterEncounter encounter) {
    static auto reverseMap = createReverseEncounterMap();
    auto it = reverseMap.find(encounter);
    if (it != reverseMap.end()) {
        return it->second;
    }
    return "UNKNOWN";
}

FightDefinition FightDefinition::fromJson(const nlohmann::json& j) {
    FightDefinition fight;
    
    if (!j.contains("fight_type")) {
        throw std::invalid_argument("JSON must contain 'fight_type' field");
    }
    
    fight.fight_type = parseMonsterEncounter(j["fight_type"]);
    fight.description = j.value("description", "");
    
    return fight;
}

FightDefinition FightDefinition::fromJsonFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    nlohmann::json j;
    file >> j;
    
    return fromJson(j);
}

}