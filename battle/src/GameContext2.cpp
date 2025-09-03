//
// Created by gamerpuppy on 6/25/2021.
//

#include "GameContext2.h"

#include <algorithm>
#include <cassert>

#include "constants/CardPools.h"
#include "BattleContext2.h"
#include "game/SaveFile.h"
#include "Game2.h"
#include "sim/PrintHelpers.h"
#include "sts_common.h"

using namespace sts;

int rollWeightedIdx(float roll, const float *weights, int weightSize);

bool isCampfireRelic(RelicId r) {
    return r == RelicId::PEACE_PIPE || r == RelicId::SHOVEL || r == RelicId::GIRYA;
}

SelectScreenCard::SelectScreenCard(const Card &card) : card(card) {}

SelectScreenCard::SelectScreenCard(const Card &card, int deckIdx) : card(card), deckIdx(deckIdx) {}

GameContext::GameContext(CharacterClass cc, std::uint64_t seed, int ascension)
    : seed(seed),
    neowRng(seed),
    treasureRng(seed),
    eventRng(seed),
    relicRng(seed),
    potionRng(seed),
    cardRng(seed),
    cardRandomRng(seed),
    monsterRng(seed),
    shuffleRng(seed),
    miscRng(seed),
    mathUtilRng(seed-897897), // uses a time based seed -_-
    cc(cc),
    ascension(ascension) {

    potionCapacity = ascension < 11 ? 3 : 2;
    std::fill(potions.begin(), potions.end(), Potion::EMPTY_POTION_SLOT);
}


int GameContext::fractionMaxHp(float percent, HpType type) const {
    if (type == ROUND) {
        return static_cast<int>(std::round(static_cast<float>(maxHp) * percent));

    } else if (type == FLOOR) {
        return static_cast<int>(static_cast<float>(maxHp) * percent);

    } else {
        return static_cast<int>(std::ceil(static_cast<float>(maxHp) * percent));
    }
}

bool GameContext::hasRelic(RelicId r) const {
    return relics.has(r);
}

void GameContext::obtainGold(int amount) {
    if (relics.has(R::ECTOPLASM)) {
        return;
    }

    gold += amount;
    if (relics.has(R::BLOODY_IDOL)) {
        playerHeal(5);
    }
}

void GameContext::obtainPotion(Potion p) {
    if (relics.has(RelicId::SOZU) || potionCount == potionCapacity) {
        return;
    }

    for (int i = 0; i < potionCapacity; ++i) {
        if (potions[i] == Potion::EMPTY_POTION_SLOT) {
            potions[i] = p;
            ++potionCount;
            return;
        }
    }

    // todo, just ignoring if there is not enough space for now
//#ifdef sts_asserts
//    assert(false);
//#endif
}

void GameContext::relicsOnEnterRoom(Room room) {
    if (hasRelic(RelicId::MAW_BANK) && relics.getRelicValue(RelicId::MAW_BANK) != 0) {
        obtainGold(12);
    }

    switch (room) {
        case Room::REST:
            if (hasRelic(RelicId::ETERNAL_FEATHER)) {
                playerHeal(deck.size() / 5 * 3);
            }
            break;

        case Room::EVENT:
            if (hasRelic(RelicId::SSSERPENT_HEAD)) {
                obtainGold(50);
            }
            break;

        default:
            break;
    }
}

CardRarity GameContext::rollCardRarity(Room room) {
    int roll = cardRng.random(99) + cardRarityFactor;

    if (room == Room::BOSS) {
        return CardRarity::RARE;
    }

    int rareChance = (room == Room::ELITE ? 10 : 3);
    const int uncommonChance = (room == Room::ELITE ? 40 : 37);

    if (room != Room::REST && hasRelic(RelicId::NLOTHS_GIFT)) {
        rareChance = rareChance * 3;
    }

    if (roll < rareChance) {
        return CardRarity::RARE;

    } else if (roll < rareChance + uncommonChance) {
        return CardRarity::UNCOMMON;

    } else {
        return CardRarity::COMMON;
    }
}

CardId GameContext::returnTrulyRandomCardFromAvailable(Random &rng, CardId exclude) {
    auto color = getCardColor(exclude);
    switch (color) {
        case CardColor::COLORLESS: {
            int idx = rng.random(static_cast<int>(colorlessCardPool.size()-2));
            if (colorlessCardPool[idx] == exclude) {
                return colorlessCardPool[idx + 1];
            } else {
                return colorlessCardPool[idx];
            }
        }

        case CardColor::CURSE: {
            return getRandomCurse(cardRng);
        }

        default: {
            const CardId* pool = TransformCardPool::getPoolForClass(cc);
            int poolSize = TransformCardPool::getPoolSizeForClass(cc);

            bool excludeInPool = cardRarities[static_cast<int>(exclude)] != CardRarity::BASIC &&
                                 static_cast<CardColor>(cc) == color;

            if (excludeInPool) {
                int idx = rng.random(poolSize-2);
                if (pool[idx] == exclude) {
                    return pool[idx+1];
                } else {
                    return pool[idx];
                }
            } else {
                return pool[rng.random(poolSize-1)];
            }
        }
    }
}

CardId GameContext::returnColorlessCard(CardRarity rarity) {
    java::Collections::shuffle(colorlessCardPool.begin(), colorlessCardPool.end(),
                               java::Random(shuffleRng.randomLong()));
    for (const auto &c : colorlessCardPool) { // todo optimize
        if (getCardRarity(c) == rarity) {
            return c;
        }
    }
    return CardId::SWIFT_STRIKE;
}

int GameContext::getRandomPlayerPotionIdx() {
    if (potionCount <= 0) {
        return -1;
    }

    fixed_list<int, 5> potionIdxs;
    for (int i = 0; i < potionCapacity; ++i) {
        if(potions[i] != Potion::EMPTY_POTION_SLOT) {
            potionIdxs.push_back(i);
        }
    }
    java::Collections::shuffle(potionIdxs.begin(), potionIdxs.end(), java::Random(miscRng.nextLong()));
    return potionIdxs[0];
}

void GameContext::playerOnDie() {
    if (hasRelic(RelicId::MARK_OF_THE_BLOOM)) {
        outcome = GameOutcome::PLAYER_LOSS;
        return;
    }

    const auto hasBark = relics.has(RelicId::SACRED_BARK);
    auto it = std::find(potions.begin(), potions.end(), Potion::FAIRY_POTION);
    if (it != potions.end()) {
        *it = Potion::EMPTY_POTION_SLOT;
        --potionCount;
        curHp = std::max(1, static_cast<int>(hasBark ? 0.6f : 0.3f * static_cast<float>(maxHp)));
        return;
    }

    if (hasRelic(RelicId::LIZARD_TAIL)) {
        auto &lizardValue = relics.getRelicValueRef(RelicId::LIZARD_TAIL);
        if (lizardValue != 0) {
            lizardValue = 0;
            curHp = std::max(1, maxHp/2);
            return;
        }
    }

    outcome = GameOutcome::PLAYER_LOSS;
}


void GameContext::playerHeal(int amount) {
    if (hasRelic(RelicId::MARK_OF_THE_BLOOM)) {
        return;
    }
    curHp = std::min(curHp + amount, maxHp);
}

void GameContext::playerIncreaseMaxHp(int amount) {
    maxHp += amount;
    playerHeal(amount);
}

void GameContext::loseGold(int amount, bool inShop) {
    if (inShop && relics.has(RelicId::MAW_BANK)) {
        relics.getRelicValueRef(RelicId::MAW_BANK) = 0;
    }
    gold = std::max(0, gold-amount);
}

void GameContext::loseMaxHp(int amount) {
    maxHp -= amount;
    curHp = std::min(curHp, maxHp);
}

void GameContext::drinkPotion(Potion p) {
    switch (p) {
        case Potion::BLOOD_POTION:
            playerHeal(fractionMaxHp(hasRelic(RelicId::SACRED_BARK) ? 0.40f : 0.20f));
            break;

        case Potion::ENTROPIC_BREW: {
            Potion randPotions[5];
            for (int i = 0 ; i < potionCapacity; ++i) {
                randPotions[i] = returnRandomPotion(potionRng, cc);
                if (potions[i] == Potion::EMPTY_POTION_SLOT) {
                    potions[i] = randPotions[i];
                }
            }
            potionCount = potionCapacity;
            break;
        }

        case Potion::FRUIT_JUICE:
            playerIncreaseMaxHp(hasRelic(RelicId::SACRED_BARK) ? 10 : 5);
            break;

        case Potion::INVALID:
        case Potion::EMPTY_POTION_SLOT:
        default:
#ifdef sts_asserts
            assert(false);
#endif
            break;
    }
}

void GameContext::drinkPotionAtIdx(int idx) {
#ifdef sts_asserts
    assert(potions[idx] != Potion::EMPTY_POTION_SLOT);
    assert(potions[idx] != Potion::INVALID);
#endif
    const Potion p = potions[idx];
    discardPotionAtIdx(idx);
    drinkPotion(p);
}

void GameContext::discardPotionAtIdx(int idx) {
#ifdef sts_asserts
    assert(potions[idx] != Potion::EMPTY_POTION_SLOT);
    assert(potions[idx] != Potion::INVALID);
#endif
    potions[idx] = Potion::EMPTY_POTION_SLOT;
    --potionCount;
}
