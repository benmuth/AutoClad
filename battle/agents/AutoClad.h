//
// AutoClad - Automated agent functions moved from SimpleAgent2
//

#ifndef STS_LIGHTSPEED_AUTOCLAD_H
#define STS_LIGHTSPEED_AUTOCLAD_H

#include <cstdint>
#include <mutex>
#include "../GameContext2.h"
#include "../BattleContext2.h"
#include "../Action2.h"
#include "../CardInstance2.h"
#include "SimpleAgent2.h"

namespace sts {

// SimpleAgentInfo struct definition
struct SimpleAgentInfo {
    bool shouldPrint;
    std::uint64_t seedStart;
    std::uint64_t seedEnd;

    std::mutex m;
    std::uint64_t curSeed;
    std::int64_t winCount = 0;
    std::int64_t lossCount = 0;
    std::int64_t floorSum = 0;
};

namespace search {

// Experimental card selection algorithm
void myGetBestCardToPlay();

// Agent runner functions
void myAgentMtRunner(SimpleAgentInfo *info);

// TODO: add name printing method!

// Simple agent with experimental methods
struct AutoClad : search::SimpleAgent {
    // static void myRunAgentMt(int threadCount, std::uint64_t startSeed, int playoutCount, bool print);
    void stepBattleCardPlay(BattleContext &bc) override;
};

} // namespace search
} // namespace sts

#endif //STS_LIGHTSPEED_AUTOCLAD_H
