import json
from os import listdir
from copy import copy
import sys
import shutil

relics = [
    "WHETSTONE",
    "THE_BOOT",
    "BLOOD_VIAL",
    "MEAL_TICKET",
    "PEN_NIB",
    "AKABEKO",
    "LANTERN",
    "REGAL_PILLOW",
    "BAG_OF_PREPARATION",
    "ANCIENT_TEA_SET",
    "SMILING_MASK",
    "POTION_BELT",
    "PRESERVED_INSECT",
    "OMAMORI",
    "MAW_BANK",
    "ART_OF_WAR",
    "TOY_ORNITHOPTER",
    "CERAMIC_FISH",
    "VAJRA",
    "CENTENNIAL_PUZZLE",
    "STRAWBERRY",
    "HAPPY_FLOWER",
    "ODDLY_SMOOTH_STONE",
    "WAR_PAINT",
    "BRONZE_SCALES",
    "JUZU_BRACELET",
    "DREAM_CATCHER",
    "NUNCHAKU",
    "TINY_CHEST",
    "ORICHALCUM",
    "ANCHOR",
    "BAG_OF_MARBLES",
    "RED_SKULL",
    "BOTTLED_TORNADO",
    "SUNDIAL",
    "KUNAI",
    "PEAR",
    "BLUE_CANDLE",
    "ETERNAL_FEATHER",
    "STRIKE_DUMMY",
    "SINGING_BOWL",
    "MATRYOSHKA",
    "INK_BOTTLE",
    "THE_COURIER",
    "FROZEN_EGG",
    "ORNAMENTAL_FAN",
    "BOTTLED_LIGHTNING",
    "GREMLIN_HORN",
    "HORN_CLEAT",
    "TOXIC_EGG",
    "LETTER_OPENER",
    "QUESTION_CARD",
    "BOTTLED_FLAME",
    "SHURIKEN",
    "MOLTEN_EGG",
    "MEAT_ON_THE_BONE",
    "DARKSTONE_PERIAPT",
    "MUMMIFIED_HAND",
    "PANTOGRAPH",
    "WHITE_BEAST_STATUE",
    "MERCURY_HOURGLASS",
    "SELF_FORMING_CLAY",
    "PAPER_PHROG",
    "GINGER",
    "OLD_COIN",
    "BIRD_FACED_URN",
    "UNCEASING_TOP",
    "TORII",
    "STONE_CALENDAR",
    "SHOVEL",
    "WING_BOOTS",
    "THREAD_AND_NEEDLE",
    "TURNIP",
    "ICE_CREAM",
    "CALIPERS",
    "LIZARD_TAIL",
    "PRAYER_WHEEL",
    "GIRYA",
    "DEAD_BRANCH",
    "DU_VU_DOLL",
    "POCKETWATCH",
    "MANGO",
    "INCENSE_BURNER",
    "GAMBLING_CHIP",
    "PEACE_PIPE",
    "CAPTAINS_WHEEL",
    "FOSSILIZED_HELIX",
    "TUNGSTEN_ROD",
    "MAGIC_FLOWER",
    "CHARONS_ASHES",
    "CHAMPION_BELT",
    "FUSION_HAMMER",
    "VELVET_CHOKER",
    "RUNIC_DOME",
    "SLAVERS_COLLAR",
    "SNECKO_EYE",
    "PANDORAS_BOX",
    "CURSED_KEY",
    "BUSTED_CROWN",
    "ECTOPLASM",
    "TINY_HOUSE",
    "SOZU",
    "PHILOSOPHERS_STONE",
    "ASTROLABE",
    "BLACK_STAR",
    "SACRED_BARK",
    "EMPTY_CAGE",
    "RUNIC_PYRAMID",
    "CALLING_BELL",
    "COFFEE_DRIPPER",
    "BLACK_BLOOD",
    "MARK_OF_PAIN",
    "RUNIC_CUB" "SLING_OF_COURAGE",
    "HAND_DRILL",
    "TOOLBOX",
    "CHEMICAL_X",
    "LEES_WAFFLE",
    "ORRERY",
    "DOLLYS_MIRROR",
    "ORANGE_PELLETS",
    "PRISMATIC_SHARD",
    "CLOCKWORK_SOUVENIR",
    "FROZEN_EYE",
    "THE_ABACUS",
    "MEDICAL_KIT",
    "CAULDRON",
    "STRANGE_SPOON",
    "MEMBERSHIP_CARD",
    "BRIMSTONE",
]

cards = [
    "ANGER",
    "CLEAVE",
    "WARCRY",
    "FLEX",
    "IRON_WAVE",
    "BODY_SLAM",
    "TRUE_GRIT",
    "SHRUG_IT_OFF",
    "CLASH",
    "THUNDERCLAP",
    "POMMEL_STRIKE",
    "TWIN_STRIKE",
    "CLOTHESLINE",
    "ARMAMENTS",
    "HAVOC",
    "HEADBUTT",
    "WILD_STRIKE",
    "HEAVY_BLADE",
    "PERFECTED_STRIKE",
    "SWORD_BOOMERANG",
    "EVOLVE",
    "UPPERCUT",
    "GHOSTLY_ARMOR",
    "FIRE_BREATHING",
    "DROPKICK",
    "CARNAGE",
    "BLOODLETTING",
    "RUPTURE",
    "SECOND_WIND",
    "SEARING_BLOW",
    "BATTLE_TRANCE",
    "SENTINEL",
    "ENTRENCH",
    "RAGE",
    "FEEL_NO_PAIN",
    "DISARM",
    "SEEING_RED",
    "DARK_EMBRACE",
    "COMBUST",
    "WHIRLWIND",
    "SEVER_SOUL",
    "RAMPAGE",
    "SHOCKWAVE",
    "METALLICIZE",
    "BURNING_PACT",
    "PUMMEL",
    "FLAME_BARRIER",
    "BLOOD_FOR_BLOOD",
    "INTIMIDATE",
    "HEMOKINESIS",
    "RECKLESS_CHARGE",
    "INFERNAL_BLADE",
    "DUAL_WIELD",
    "POWER_THROUGH",
    "INFLAME",
    "SPOT_WEAKNESS",
    "DOUBLE_TAP",
    "DEMON_FORM",
    "BLUDGEON",
    "FEED",
    "LIMIT_BREAK",
    "CORRUPTION",
    "BARRICADE",
    "FIEND_FIRE",
    "BERSERK",
    "IMPERVIOUS",
    "JUGGERNAUT",
    "BRUTALITY",
    "REAPER",
    "EXHUME",
    "OFFERING",
    "IMMOLATE",
]


class Scenario:
    def __init__(self, floor, deck) -> None:
        self.floor = floor
        self.deck = deck
        # map of relics to their counters (None if they don't have a counter)
        self.relics = []
        self.max_hp = 0
        self.current_hp = 0
        self.encounter = None
        self.potions = (None, None, None, None, None)


# runs = [f for f in listdir("2019-runs-json")]

# dir = input("enter a directory name: ")

initial_deck = [
    "STRIKE",
    "STRIKE",
    "STRIKE",
    "STRIKE",
    "STRIKE",
    "DEFEND",
    "DEFEND",
    "DEFEND",
    "DEFEND",
    "BASH",
]

dir = sys.argv[1]

for file_name in listdir(dir):
    # for each file describing a run, we're creating a list of derived scenarios
    run_scenarios = []

    with open(dir + file_name, "r") as f:
        print(file_name)
        content = json.load(f)

        campfire_choices = content["campfire_choices"]
        character_chosen = content["character_chosen"]
        items_purchased = content["items_purchased"]
        item_purchase_floors = content["item_purchase_floors"]
        current_hp_per_floor = content["current_hp_per_floor"]
        max_hp_per_floor = content["max_hp_per_floor"]
        card_choices = content["card_choices"]
        relics_obtained = content["relics_obtained"]
        event_choices = content["event_choices"]
        boss_relics = content["boss_relics"]

        if character_chosen != "IRONCLAD":
            print(f"skipping {character_chosen} run!")
            continue

        for choice in card_choices:
            if len(run_scenarios) > 0:
                deck = copy(run_scenarios[-1].deck)
            else:
                deck = copy(initial_deck)

            scenario = Scenario(choice["floor"], deck)
            run_scenarios.append(scenario)
            deck.append(choice["picked"])

        for event in event_choices:
            if "cards_removed" in event:
                for scenario in run_scenarios:
                    if event["floor"] < scenario.floor:
                        for card_removed in event["cards_removed"]:
                            if "strike" in card_removed.lower():
                                print("removing STRIKE")
                                scenario.deck.remove("STRIKE")
                            elif "defend " in card_removed.lower():
                                print("removing DEFEND")
                                scenario.deck.remove("DEFEND")
                            else:
                                print(f"removing {card_removed}")
                                scenario.deck.remove(card_removed)

            if "cards_obtained" in event:
                for scenario in run_scenarios:
                    if event["floor"] < scenario.floor:
                        for card_obtained in event["cards_obtained"]:
                            print(f"adding {card_obtained}")
                            scenario.deck.append(card_obtained)

        for relic in relics_obtained:
            floor = relic["floor"]
            for scenario in run_scenarios:
                if scenario.floor > floor:
                    scenario.relics.append(relic["key"])

        for purchase in zip(item_purchase_floors, items_purchased):
            for scenario in run_scenarios:
                if scenario.floor > purchase[0]:
                    scenario.deck.append(purchase[1])

            # print("purchase", purchase)

        # for scenario in run_scenarios:
        #     print(scenario.deck)
        #     print(scenario.relics)

        print('')
        print('last scenario deck', sorted(sorted(run_scenarios[-1].deck)))
        print('')
        print('master scenario deck', sorted(sorted(content["master_deck"])))
        print('\n')
        print('last scenario relics', sorted(sorted(run_scenarios[-1].relics)))
        print('')
        print('master scenario relics', sorted(sorted(content["relics"])))
        print('\n')

        assert sorted(run_scenarios[-1].deck) == sorted(content["master_deck"])

        break
