//
// Created by keega on 9/27/2021.
//

#ifndef STS_LIGHTSPEED_SIMPLEAGENT_H
#define STS_LIGHTSPEED_SIMPLEAGENT_H

#include "GameContext2.h"
#include "Action2.h"
// #include "sim/search/GameAction.h"

// Forward declarations to avoid conflicts
namespace sts {
    class BattleContext;
}

namespace sts::search {

    struct SimpleAgent {

        std::vector<int> actionHistory;
        GameContext *curGameContext; // unsafe only use in private methods during playout

        fixed_list<int,16> mapPath;

        bool print = false;

        SimpleAgent();

        [[nodiscard]] int getIncomingDamage(const BattleContext &bc) const;

        // void playout(GameContext &gc);

        // void takeAction(GameContext &gc, GameAction a);
        void takeAction(BattleContext &bc, Action a);
        void playoutBattle(BattleContext &bc, std::stringstream* snapshot = nullptr);

        // stepBattleCardPlay is the function that chooses the card to play
        virtual void stepBattleCardPlay(BattleContext &bc);
        void stepBattleCardSelect(BattleContext &bc);

        bool playPotion(BattleContext &bc);
        static void runAgentsMt(int threadCount, std::uint64_t startSeed, int playoutCount, bool print);

        // Get action sequence as a simple list of action descriptions
        std::vector<std::string> getActionSequence() const;
    };

}

#endif //STS_LIGHTSPEED_SIMPLEAGENT_H
