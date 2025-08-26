import json
from os import listdir
from copy import copy
import sys

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
    "CULTIST_HEADPIECE",
    "FACE_OF_CLERIC",
    "GREMLIN_VISAGE",
    "NLOTHS_HUNGRY_FACE",
    "SSSERPENT_HEAD",
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


def ensure_scenario_after_floor(floor, run_scenarios, initial_deck):
    """
    Ensures there's at least one scenario after the given floor.
    If not, creates a new scenario and adds it to run_scenarios.
    Returns list of scenarios that occur after the floor.
    """
    scenarios_after_floor = [s for s in run_scenarios if s.floor > floor]

    if not scenarios_after_floor:
        if run_scenarios:
            new_deck = run_scenarios[-1].deck.copy()
            new_scenario = Scenario(floor + 1, new_deck)
            new_scenario.relics = run_scenarios[-1].relics.copy()
        else:
            new_scenario = Scenario(floor + 1, copy(initial_deck))
            new_scenario.relics.append("Burning Blood")

        run_scenarios.append(new_scenario)
        scenarios_after_floor = [new_scenario]

    return scenarios_after_floor


initial_deck = [
    "AscendersBane",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Strike_R",
    "Defend_R",
    "Defend_R",
    "Defend_R",
    "Defend_R",
    "Bash",
]

dir = sys.argv[1]

num = 0
for file_name in listdir(dir):
    # for each file describing a run, we're creating a list of derived scenarios
    run_scenarios = []

    with open(dir + file_name, "r") as f:
        print(file_name)
        content = json.load(f)

        if content["ascension_level"] < 15:
            print(f"skipping ascension level {content["ascension_level"]} run!")
            continue

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
        items_purged = content["items_purged"]
        items_purged_floors = content["items_purged_floors"]
        neow_bonus = content.get("neow_bonus", "")
        floor_reached = content["floor_reached"]
        if floor_reached < 5:
            continue

        if character_chosen != "IRONCLAD":
            print(f"skipping {character_chosen} run!")
            continue

        for choice in card_choices:
            if len(run_scenarios) > 0:
                deck = copy(run_scenarios[-1].deck)
            else:
                deck = copy(initial_deck)

            scenario = Scenario(choice["floor"], deck)

            # Add starting relic for Ironclad
            if len(run_scenarios) == 0:  # First scenario
                scenario.relics.append("Burning Blood")
            else:
                scenario.relics = run_scenarios[-1].relics.copy()
            run_scenarios.append(scenario)
            if choice["picked"] != "SKIP":
                deck.append(choice["picked"])

        boss_floors = [16, 33, 50]
        for i, boss_relic in enumerate(boss_relics):
            picked_relic = boss_relic["picked"]
            boss_floor = boss_floors[i] if i < len(boss_floors) else boss_floors[-1]
            for scenario in run_scenarios:
                if scenario.floor > boss_floor:
                    scenario.relics.append(picked_relic)

        for event in event_choices:
            if "cards_obtained" in event:
                scenarios_after_floor = ensure_scenario_after_floor(
                    event["floor"], run_scenarios, initial_deck
                )
                for scenario in scenarios_after_floor:
                    for card_obtained in event["cards_obtained"]:
                        scenario.deck.append(card_obtained)

            if "cards_upgraded" in event:
                scenarios_after_floor = ensure_scenario_after_floor(
                    event["floor"], run_scenarios, initial_deck
                )
                for scenario in scenarios_after_floor:
                    for card_upgraded in event["cards_upgraded"]:
                        if card_upgraded in scenario.deck:
                            idx = scenario.deck.index(card_upgraded)
                            scenario.deck[idx] = card_upgraded + "+1"
                        elif card_upgraded + "+1" in scenario.deck:
                            idx = scenario.deck.index(card_upgraded + "+1")
                            scenario.deck[idx] = card_upgraded + "+2"

            if "relics_obtained" in event:
                scenarios_after_floor = ensure_scenario_after_floor(
                    event["floor"], run_scenarios, initial_deck
                )
                for scenario in scenarios_after_floor:
                    for relic_obtained in event["relics_obtained"]:
                        scenario.relics.append(relic_obtained)

        face_trader_relics = [
            "Cultist Headpiece",
            "Face of Cleric",
            "Gremlin Visage",
            "N'loth's Hungry Face",
            "Ssserpent Head",
        ]

        for event in event_choices:
            if (
                event.get("event_name") == "FaceTrader"
                and event.get("player_choice") == "Trade"
            ):
                floor = event["floor"]
                for face_relic in face_trader_relics:
                    if face_relic in content["relics"]:
                        scenarios_after_floor = ensure_scenario_after_floor(
                            floor, run_scenarios, initial_deck
                        )
                        for scenario in scenarios_after_floor:
                            scenario.relics.append(face_relic)
                        break

        for relic in relics_obtained:
            floor = relic["floor"]
            scenarios_after_floor = ensure_scenario_after_floor(
                floor - 1, run_scenarios, initial_deck
            )
            for scenario in scenarios_after_floor:
                scenario.relics.append(relic["key"])

        for purchase in zip(item_purchase_floors, items_purchased):
            floor = purchase[0]
            item_name = purchase[1]

            scenarios_after_floor = ensure_scenario_after_floor(
                floor, run_scenarios, initial_deck
            )

            for scenario in scenarios_after_floor:
                if item_name.replace(" ", "").replace("_", "").lower() in [
                    r.replace(" ", "").replace("_", "").lower()
                    for r in content["relics"]
                ]:
                    scenario.relics.append(item_name)
                elif "Potion" in item_name:
                    continue
                else:
                    scenario.deck.append(item_name)

        for campfire in campfire_choices:
            if campfire["key"] == "SMITH" and "data" in campfire:
                card_name = campfire["data"]
                floor = campfire["floor"]
                for scenario in run_scenarios:
                    if scenario.floor > floor:
                        # Find the card in deck and upgrade it
                        if card_name in scenario.deck:
                            idx = scenario.deck.index(card_name)
                            scenario.deck[idx] = card_name + "+1"
                        elif card_name + "+1" in scenario.deck:
                            idx = scenario.deck.index(card_name + "+1")
                            scenario.deck[idx] = card_name + "+2"

        if "RELIC" in neow_bonus.upper():
            master_relics = set(content["relics"])
            derived_relics = set(run_scenarios[-1].relics)
            missing_relics = master_relics - derived_relics

            if len(missing_relics) == 1:
                neow_relic = missing_relics.pop()
                for scenario in run_scenarios:
                    scenario.relics.append(neow_relic)
        elif neow_bonus == "THREE_ENEMY_KILL":
            for scenario in run_scenarios:
                scenario.relics.append("NeowsBlessing")

        # handle removal last so we don't try to remove non-existing items
        for event in event_choices:
            if "cards_removed" in event:
                for scenario in run_scenarios:
                    if event["floor"] < scenario.floor:
                        for card_removed in event["cards_removed"]:
                            if "strike_r+1" in card_removed.lower():
                                if "Strike_R" in scenario.deck:
                                    scenario.deck.remove("Strike_R")
                                elif "Strike_R+1" in scenario.deck:
                                    scenario.deck.remove("Strike_R+1")
                            elif "defend_r+1" in card_removed.lower():
                                if "Defend_R" in scenario.deck:
                                    scenario.deck.remove("Defend_R")
                                elif "Defend_R+1" in scenario.deck:
                                    scenario.deck.remove("Defend_R+1")
                            else:
                                if card_removed in scenario.deck:
                                    scenario.deck.remove(card_removed)

            if "relics_lost" in event:
                print(event)
                scenarios_after_floor = ensure_scenario_after_floor(
                    event["floor"], run_scenarios, initial_deck
                )
                for scenario in scenarios_after_floor:
                    for relic_lost in event["relics_lost"]:
                        print("LOST: ", relic_lost)
                        print("relics: ", scenario.relics)

                        scenario.relics.remove(relic_lost)

        for purge in zip(items_purged_floors, items_purged):
            card_name = purge[1]
            floor = purge[0]
            scenarios_after_floor = ensure_scenario_after_floor(
                floor, run_scenarios, initial_deck
            )
            for scenario in scenarios_after_floor:
                if card_name in scenario.deck:
                    scenario.deck.remove(card_name)

        print("NUM: ", num)
        print(f"Number of scenarios: {len(run_scenarios)}")
        print(f"Last scenario floor: {run_scenarios[-1].floor}")
        print("Scenario floors:", [s.floor for s in run_scenarios])
        print(f"Neow bonus: {neow_bonus}")

        print("")
        print("last scenario deck  ", sorted(sorted(run_scenarios[-1].deck)))
        print("")
        print("master scenario deck", sorted(sorted(content["master_deck"])))
        print("\n")
        print("last scenario relics  ", sorted(sorted(run_scenarios[-1].relics)))
        print("")
        print("master scenario relics", sorted(sorted(content["relics"])))
        print("\n")
        print(
            "scenarios: ",
        )
        # assert sorted(run_scenarios[-1].deck) == sorted(content["master_deck"])
        assert sorted(run_scenarios[-1].relics) == sorted(content["relics"])
        num += 1
