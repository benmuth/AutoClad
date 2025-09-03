//
// Created by gamerpuppy on 6/24/2021.
//

#ifndef STS_LIGHTSPEED_GAMECONTEXT_H
#define STS_LIGHTSPEED_GAMECONTEXT_H

#include <array>
#include <functional>

#include "data_structure/fixed_list.h"

#include "constants/Misc.h"
#include "constants/CardPools.h"
#include "constants/Potions.h"
#include "constants/Relics.h"
#include "constants/Rooms.h"
#include "constants/MonsterEncounters.h"
#include "constants/CharacterClasses.h"

#include "game/Random.h"
#include "game/RelicContainer.h"
#include "game/Card.h"
#include "Deck2.h"

namespace sts {

    enum class GameOutcome {
        PLAYER_LOSS,
        UNDECIDED,
        PLAYER_VICTORY,
    };

    enum RngReference {
        MISC_RNG,
        CARD_RNG,
        NEOW_RNG,
    };

    enum class CardSelectScreenType {
        INVALID=0,
        TRANSFORM,
        TRANSFORM_UPGRADE,
        UPGRADE,
        REMOVE,
        DUPLICATE,
        OBTAIN,
        BOTTLE,
        BONFIRE_SPIRITS,
    };

    enum class ScreenState {
        INVALID=0,
        CARD_SELECT,
        BATTLE,
    };

    struct SelectScreenCard {
        Card card;
        std::int16_t deckIdx = -1;

        SelectScreenCard() = default;
        SelectScreenCard(const Card &card);
        SelectScreenCard(const Card &card, int deckIdx);
    };

    struct ScreenStateInfo {
        MonsterEncounter encounter;

        // CardSelectScreen
        RngReference transformRng = CARD_RNG;
        CardSelectScreenType selectScreenType = CardSelectScreenType::INVALID;
        int toSelectCount = 0;
        fixed_list<SelectScreenCard, Deck::MAX_SIZE> toSelectCards;
        fixed_list<SelectScreenCard,3> haveSelectedCards;

        // from combats
        int stolenGold = 0;
    };


    class GameContext;
    typedef std::function<void(GameContext&)> GameContextAction;

    class BattleContext;
    class SaveFile;

    struct GameContext {
        static constexpr float SHRINE_CHANCE = 0.25F;

        sts::Card noteForYourselfCard = Card(CardId::IRON_WAVE);

        static constexpr bool disableColosseum = true;
        static constexpr bool disableMatchAndKeep = true;
        static constexpr bool disablePrismaticShard = true;
        bool skipBattles = false;

        // ********* hidden from player *********
        std::uint64_t seed;

        Random aiRng;
        Random cardRandomRng;
        Random cardRng;
        Random eventRng;
        Random mathUtilRng;
        Random merchantRng;
        Random miscRng;
        Random monsterHpRng;
        Random monsterRng;
        Random neowRng;
        Random potionRng;
        Random relicRng;
        Random shuffleRng;
        Random treasureRng;

        std::array<CardId, 35> colorlessCardPool = baseColorlessPool;

        // ********* player information *********

        GameOutcome outcome = GameOutcome::UNDECIDED;
        ScreenState screenState = ScreenState::INVALID;
        ScreenStateInfo info;

        Room lastRoom = Room::INVALID;
        Room curRoom = Room::INVALID;
        MonsterEncounter boss = MonsterEncounter::INVALID;

        int cardRarityFactor = 5;

        int act = 1;
        int ascension = 0;
        int floorNum = 0;

        CharacterClass cc;
        int curHp = 80;
        int maxHp = 80;
        int gold = 99;

        int potionCount = 0;
        int potionCapacity = 0;
        std::array<Potion, 5> potions;

        RelicContainer relics;
        Deck deck;

        GameContextAction regainControlAction = nullptr;

        GameContext() = default;
        GameContext(CharacterClass cc, std::uint64_t seed, int ascensionLevel);

        void initFromSave(const SaveFile &s);
        void initRelicsFromSave(const SaveFile &s);

        // const methods
        [[nodiscard]] int fractionMaxHp(float percent, HpType type=HpType::FLOOR) const;
        [[nodiscard]] bool hasRelic(RelicId r) const;

        // room setup
        void enterBattle(MonsterEncounter encounter);
        void afterBattle();

        // actions
        void obtainCard(Card c, int count=1);
        void obtainGold(int amount);
        void obtainPotion(Potion p);

        void relicsOnEnterRoom(Room room);

        CardRarity rollCardRarity(Room room);
        CardId returnTrulyRandomCardFromAvailable(Random &rng, CardId exclude);

        Card getTransformedCard(Random &rng, CardId exclude, bool autoUpgrade= false);

        CardId returnColorlessCard(CardRarity rarity);
        int getRandomPlayerPotionIdx();
        int getRandomPlayerNonBasicCardIdx();

        // actions
        void damagePlayer(int amount);
        void playerLoseHp(int amount);
        void playerOnDie();
        void playerHeal(int amount);
        void playerIncreaseMaxHp(int amount);
        void loseGold(int amount, bool inShop=false);
        void loseMaxHp(int amount);


        void drinkPotion(Potion p);
        void drinkPotionAtIdx(int idx);
        void discardPotionAtIdx(int idx);


        void openCardSelectScreen(CardSelectScreenType type, int selectCount, bool initSelectCards=true);

        // interface methods
        void chooseSelectCardScreenOption(int idx);

        void regainControl();
    };


}



#endif //STS_LIGHTSPEED_GAMECONTEXT_H
