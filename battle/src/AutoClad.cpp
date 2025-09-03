//
// AutoClad - Automated agent functions moved from SimpleAgent2
//

#include <algorithm>
#include <iostream>
#include <deque>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

#include "AutoClad.h"
#include "GameContext2.h"
#include "BattleContext2.h"
#include "SimpleAgent2.h"
#include "../../include/constants/CharacterClasses.h"
#include "../../include/constants/MonsterEncounters.h"

using namespace sts;

// State struct for myGetBestCardToPlay
typedef struct {
    BattleContext bc;
    CardInstance card;
} State;

void search::AutoClad::stepBattleCardPlay(BattleContext &bc)  {
    takeAction(bc, Action(ActionType::END_TURN));
    // std::cerr << ""
    // takeAction(bc, Action(ActionType::CARD, 0));

    // GameContext gc = GameContext(CharacterClass::IRONCLAD, 0, 0);
    // bc.init(gc, MonsterEncounter::JAW_WORM, false);

    // int seed_sample_count = 10;

    // enumerate actions
    // std::deque<State> states;
    // for (int i = 0; i < std::size(bc.cards.hand); ++i) {
    //     if (bc.cards.hand[i].id == CardId::INVALID) break;
    //     auto new_bc = bc;
    //     // auto action = search::Action(sts::search::ActionType::CARD, i);

    //     State state = {new_bc, bc.cards.hand[i]};
    //     states.push_front(state);
    // }
    // sort actions by heuristic

    // for (int i = 0; i < seed_sample_count; ++i) {

    //     // start dfs with first action
    //     while (!states.empty()) {
    //         auto state = states.front();
    //         states.pop_front();

    //         std::cout << "state: " << (state.card.getName()) << "\n";
    //         std::cout << "hand: ";
    //         bool first = true;
    //         for (const auto& element : state.bc.cards.hand) {
    //             if (!first) std::cout << ", ";
    //             std::cout << element;
    //             first = false;
    //         }
    //         std::cout << "\n";
    //         // execute action
    //         if (!state.card.requiresTarget()) {
    //             auto *bestCard = std::find(std::begin(state.bc.cards.hand), std::end(state.bc.cards.hand), &state.card);
    //             auto bestCardIdx = std::distance(std::begin(state.bc.cards.hand), bestCard);

    //             auto action = search::Action(sts::search::ActionType::CARD, bestCardIdx);
    //             action.execute(state.bc);
    //         }

    //         // push new states
    //         for (int i = 0; i < std::size(state.bc.cards.hand); ++i) {
    //             if (state.bc.cards.hand[i].id == CardId::INVALID) break;
    //             auto new_bc = state.bc;
    //             // auto action = search::Action(sts::search::ActionType::CARD, i);

    //             State state = {new_bc, state.bc.cards.hand[i]};
    //             states.push_front(state);
    //         }

    //         // eval action
    //     }
    // }

    // __dfs__ (with pruning)
}

void search::myAgentMtRunner(SimpleAgentInfo *info) {
    GameContext gc(CharacterClass::IRONCLAD, info->curSeed, 0);
    BattleContext bc;
    bc.init(gc, MonsterEncounter::JAW_WORM, false);

    search::SimpleAgent agent;
    agent.print = info->shouldPrint;
    
    if (info->shouldPrint) {
        std::cout << "Starting battle with seed " << info->curSeed << std::endl;
        std::cout << "Player HP: " << bc.player.curHp << "/" << bc.player.maxHp << std::endl;
    }
    
    agent.playoutBattle(bc);
    
    if (info->shouldPrint) {
        std::cout << "Battle finished!" << std::endl;
        std::cout << "Outcome: ";
        switch (bc.outcome) {
            case Outcome::PLAYER_VICTORY:
                std::cout << "PLAYER_VICTORY" << std::endl;
                break;
            case Outcome::PLAYER_LOSS:
                std::cout << "PLAYER_LOSS" << std::endl;
                break;
            default:
                std::cout << "UNDECIDED" << std::endl;
                break;
        }
        std::cout << "Final HP: " << bc.player.curHp << "/" << bc.player.maxHp << std::endl;
        std::cout << "Turns: " << bc.turn << std::endl;
    }
}

// void search::AutoClad::myRunAgentMt(int threadCount, std::uint64_t startSeed, int playoutCount, bool print) {
//     SimpleAgentInfo info;
//     info.curSeed = startSeed;
//     info.seedStart = startSeed;
//     info.seedEnd = startSeed + playoutCount;
//     info.shouldPrint = print;

//     myAgentMtRunner(&info);
// }
