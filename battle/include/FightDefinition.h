#ifndef BATTLE_FIGHT_DEFINITION_H
#define BATTLE_FIGHT_DEFINITION_H

#include <string>
#include <nlohmann/json.hpp>
#include "constants/MonsterEncounters.h"

namespace battle {

struct FightDefinition {
    sts::MonsterEncounter fight_type;
    std::string description;

    static FightDefinition fromJson(const nlohmann::json& j);
    static FightDefinition fromJsonFile(const std::string& filename);
};

sts::MonsterEncounter parseMonsterEncounter(const std::string& encounter_str);
std::string monsterEncounterToString(sts::MonsterEncounter encounter);

}

#endif