import json
import os
from os import listdir
from copy import copy
import sys
import random

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

# Encounter pools by floor range
act_one_easy = ["CULTIST", "JAW_WORM", "TWO_LOUSE", "SMALL_SLIMES"]
act_one = [
    "BLUE_SLAVER",
    "GREMLIN_GANG",
    "LOOTER",
    "LARGE_SLIME",
    "LOTS_OF_SLIMES",
    "EXORDIUM_THUGS",
    "EXORDIUM_WILDLIFE",
    "RED_SLAVER",
    "THREE_LOUSE",
    "TWO_FUNGI_BEASTS",
]
act_one_elites = [
    "GREMLIN_NOB",
    "LAGAVULIN",
    "THREE_SENTRIES",
]
act_one_bosses = ["SLIME_BOSS", "THE_GUARDIAN", "HEXAGHOST"]

act_two_easy = [
    "SPHERIC_GUARDIAN",
    "CHOSEN",
    "SHELL_PARASITE",
    "THREE_BYRDS",
    "TWO_THIEVES",
]
act_two = [
    "CHOSEN_AND_BYRDS",
    "SENTRY_AND_SPHERE",
    "SNAKE_PLANT",
    "SNECKO",
    "CENTURION_AND_HEALER",
    "CULTIST_AND_CHOSEN",
    "THREE_CULTIST",
    "SHELLED_PARASITE_AND_FUNGI",
]
act_two_elites = ["GREMLIN_LEADER", "SLAVERS", "BOOK_OF_STABBING"]
act_two_bosses = ["AUTOMATON", "COLLECTOR", "CHAMP"]

act_three_easy = ["THREE_DARKLINGS", "ORB_WALKER", "THREE_SHAPES"]
act_three = [
    "SPIRE_GROWTH",
    "TRANSIENT",
    "FOUR_SHAPES",
    "MAW",
    "SPHERE_AND_TWO_SHAPES",
    "JAW_WORM_HORDE",
    "WRITHING_MASS",
]
act_three_elites = ["GREMLIN_LEADER", "SLAVERS", "BOOK_OF_STABBING"]
act_three_bosses = ["AUTOMATON", "COLLECTOR", "CHAMP"]

# Potion pools
common_potions = [
    "ATTACK_POTION",
    "BLESSING_OF_THE_FORGE",
    "BLOCK_POTION",
    "BLOOD_POTION",
    "COLORLESS_POTION",
    "DEXTERITY_POTION",
    "ENERGY_POTION",
    "EXPLOSIVE_POTION",
    "FEAR_POTION",
    "FIRE_POTION",
    "FLEX_POTION",
    "FOCUS_POTION",
    "POISON_POTION",
    "POWER_POTION",
    "SKILL_POTION",
    "SPEED_POTION",
    "STRENGTH_POTION",
    "SWIFT_POTION",
    "WEAK_POTION",
]

uncommon_potions = [
    "ANCIENT_POTION",
    "DISTILLED_CHAOS",
    "DUPLICATION_POTION",
    "ELIXIR_POTION",
    "ESSENCE_OF_STEEL",
    "GAMBLERS_BREW",
    "LIQUID_BRONZE",
    "LIQUID_MEMORIES",
    "REGEN_POTION",
]

rare_potions = [
    "CULTIST_POTION"
    "ENTROPIC_BREW"
    "FAIRY_POTION"
    "FRUIT_JUICE"
    "HEART_OF_IRON"
    "SMOKE_BOMB"
    "SNECKO_OIL"
]

# Relics that have counters with their ranges
relic_counter_ranges = {
    "HAPPY_FLOWER": (0, 3),
    "INCENSE_BURNER": (0, 6),
    "SUNDIAL": (0, 3),
    "NUNCHAKU": (0, 10),
    "PEN_NIB": (0, 10),
    "INK_BOTTLE": (0, 10),
    "LETTER_OPENER": (0, 3),
    "SHURIKEN": (0, 3),
    "KUNAI": (0, 3),
    "ORNAMENTAL_FAN": (0, 3),
    "STONE_CALENDAR": (0, 7),
    "POCKETWATCH": (0, 3),
}


def get_encounter_for_floor(floor):
    floors_per_act = 17
    act = floor // floors_per_act
    floor_within_act = floor - (act * floors_per_act)

    # print(f"act: {act}, floor: {floor_within_act}")
    match (act + 1, floor_within_act):
        case (1, f) if f < 4:
            return random.choice(act_one_easy)
        case (1, f) if f < floors_per_act // 2:
            return random.choice(act_one_easy + act_one + act_one_elites)
        case (1, f):
            return random.choice(act_one + act_one_elites)
        case (1, 17):
            return random.choice(act_one_bosses)
        case (2, f) if f < 3:
            return random.choice(act_two_easy)
        case (2, f) if f < floors_per_act // 2:
            return random.choice(act_two_easy + act_two + act_two_elites)
        case (2, f):
            return random.choice(act_two + act_two_elites)
        case (2, 17):
            return random.choice(act_two_bosses)
        case (3, f) if f < 3:
            return random.choice(act_three_easy)
        case (3, f) if f < floors_per_act // 2:
            return random.choice(act_three_easy + act_three + act_three_elites)
        case (3, f):
            return random.choice(act_three + act_three_elites)
        case (3, 17):
            return random.choice(act_three_bosses)


def generate_potions():
    num_potions = random.randint(0, 2)
    potions = []

    for _ in range(num_potions):
        rarity = random.choices(["common", "uncommon", "rare"], weights=[70, 25, 5])[0]
        if rarity == "common":
            potions.append(random.choice(common_potions))
        elif rarity == "uncommon":
            potions.append(random.choice(uncommon_potions))
        else:
            potions.append(random.choice(rare_potions))

    return potions


def generate_relic_counters(relics):
    """Generate relic counters for relics that have them."""
    counters = {}
    for relic in relics:
        # Convert relic name to match the relic_counter_ranges keys
        relic_key = relic.upper().replace(" ", "_").replace("'", "")
        if relic_key == "NLOTHSHUNGRY_FACE":
            relic_key = "NLOTHS_HUNGRY_FACE"

        if relic_key in relic_counter_ranges:
            min_val, max_val = relic_counter_ranges[relic_key]
            counters[relic_key] = random.randint(min_val, max_val - 1)
    return counters


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
    # "AscendersBane",
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


def create_scenario_file(scenario, output_dir, run_filename, scenario_index):
    """Create a scenario JSON file from a scenario object."""
    max_hp = random.randint(30, 200)
    player_hp = random.randint(1, max_hp)
    encounter = get_encounter_for_floor(scenario.floor)
    potions = generate_potions()
    relic_counters = generate_relic_counters(scenario.relics)

    # Convert deck from internal format to scenario format
    converted_deck = []
    for card in scenario.deck:
        if card == "Strike_R":
            converted_deck.append("STRIKE")
        elif card == "Strike_R+1":
            converted_deck.append("STRIKE+")
        elif card == "Strike_R+2":
            converted_deck.append("STRIKE+2")
        elif card == "Defend_R":
            converted_deck.append("DEFEND")
        elif card == "Defend_R+1":
            converted_deck.append("DEFEND+")
        elif card == "Defend_R+2":
            converted_deck.append("DEFEND+2")
        elif card == "AscendersBane":
            converted_deck.append("ASCENDERS_BANE")
        elif card.endswith("+1"):
            converted_deck.append(card[:-2].upper().replace(" ", "_") + "+")
        elif card.endswith("+2"):
            converted_deck.append(card[:-2].upper().replace(" ", "_") + "+2")
        else:
            converted_deck.append(card.upper().replace(" ", "_"))

    # Convert relics to uppercase and replace spaces/apostrophes with underscores
    converted_relics = []
    for relic in scenario.relics:
        converted_relic = relic.upper().replace(" ", "_").replace("'", "")
        # Handle special cases
        if converted_relic == "BURNING_BLOOD":
            converted_relics.append("BURNING_BLOOD")
        elif converted_relic == "CHARONS_ASHES":
            converted_relics.append("CHARONS_ASHES")
        elif converted_relic == "NLOTHSHUNGRY_FACE":
            converted_relics.append("NLOTHS_HUNGRY_FACE")
        else:
            converted_relics.append(converted_relic)

    # Extract run number from filename (e.g., "1546376628.json" -> "1546376628")
    run_number = run_filename.replace(".json", "")

    scenario_data = {
        "name": f"Generated Run {run_number} Floor {scenario.floor}",
        "description": f"Auto-generated scenario from run data (run {run_number}, scenario {scenario_index})",
        "floor": scenario.floor,
        "seed": 12345,
        "ascension": 0,
        "initial_state": {
            "player_hp": player_hp,
            "player_max_hp": max_hp,
            "character_class": "IRONCLAD",
            "encounter": encounter,
            "deck": converted_deck,
            "relics": converted_relics,
            "potions": potions,
        },
    }

    if relic_counters:
        scenario_data["initial_state"]["relic_counters"] = relic_counters

    filename = f"generated_run{run_number}_floor{scenario.floor}_{scenario_index}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(scenario_data, f, indent=2)

    return filename


def clear_output_directory(output_dir, force=False):
    """Clear all JSON files from output directory with user confirmation."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return

    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    if json_files:
        if force:
            for f in json_files:
                os.remove(os.path.join(output_dir, f))
            print(f"Deleted {len(json_files)} files.")
        else:
            print(f"Found {len(json_files)} JSON files in {output_dir}:")
            for f in json_files[:5]:  # Show first 5 files
                print(f"  {f}")
            if len(json_files) > 5:
                print(f"  ... and {len(json_files) - 5} more")

            response = input(f"\nDelete all {len(json_files)} JSON files? (y/N): ")
            if response.lower() == "y":
                for f in json_files:
                    os.remove(os.path.join(output_dir, f))
                print(f"Deleted {len(json_files)} files.")
            else:
                print("Cancelled. Files not deleted.")
                sys.exit(0)


if len(sys.argv) < 2:
    print("Usage: python scenarios-from-runs.py <input_dir> [output_dir] [--force]")
    sys.exit(1)

dir = sys.argv[1]
output_dir = (
    sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
)
force_delete = "--force" in sys.argv

if output_dir:
    clear_output_directory(output_dir, force_delete)

# num = 0
for file_name in listdir(dir):
    try:
        # for each file describing a run, we're creating a list of derived scenarios
        run_scenarios = []

        with open(dir + file_name, "r") as f:
            # print(file_name)
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
            relics = content["relics"]
            if floor_reached < 5:
                continue

            if character_chosen != "IRONCLAD":
                print(f"skipping {character_chosen} run!")
                continue

            if any("prism" in r.lower() for r in relics):
                print("skipping PrismaticShard run!")
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
                    # print(event)
                    scenarios_after_floor = ensure_scenario_after_floor(
                        event["floor"], run_scenarios, initial_deck
                    )
                    for scenario in scenarios_after_floor:
                        for relic_lost in event["relics_lost"]:
                            # print("LOST: ", relic_lost)
                            # print("relics: ", scenario.relics)

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

            # print("NUM: ", num)
            # print(f"Number of scenarios: {len(run_scenarios)}")
            # print(f"Last scenario floor: {run_scenarios[-1].floor}")
            # print("Scenario floors:", [s.floor for s in run_scenarios])
            # print(f"Neow bonus: {neow_bonus}")

            # print("")
            # print("last scenario deck  ", sorted(sorted(run_scenarios[-1].deck)))
            # print("")
            # print("master scenario deck", sorted(sorted(content["master_deck"])))
            # print("\n")
            # print("last scenario relics  ", sorted(sorted(run_scenarios[-1].relics)))
            # print("")
            # print("master scenario relics", sorted(sorted(content["relics"])))
            # print("\n")
            # print(
            #     "scenarios: ",
            # )
            # assert sorted(run_scenarios[-1].deck) == sorted(content["master_deck"])
            assert sorted(run_scenarios[-1].relics) == sorted(content["relics"])

            # Create scenario files if output directory is specified
            if output_dir:
                created_files = []
                for scenario_index, scenario in enumerate(run_scenarios):
                    filename = create_scenario_file(
                        scenario, output_dir, file_name, scenario_index
                    )
                    created_files.append(filename)
                print(f"Created {len(created_files)} scenario files for this run")

            # num += 1

    except Exception as e:
        print(f"error: {e}")
        continue
