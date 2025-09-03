#include <iostream>
#include "BattleContext2.h"
#include "GameContext2.h"
#include "FightDefinition.h"

using namespace sts;

int main() {
    try {
        // Load fight definition from JSON
        battle::FightDefinition fight = battle::FightDefinition::fromJsonFile("battle/sample_fight.json");
        
        std::cout << "Loading fight: " << fight.description << std::endl;
        std::cout << "Monster encounter: " << battle::monsterEncounterToString(fight.fight_type) << std::endl;
        
        int seed = 1984;
        GameContext gc(CharacterClass::IRONCLAD, seed, 0);

        BattleContext bc;
        bc.init(gc, fight.fight_type, false);
        
        std::cout << "Battle initialized successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

