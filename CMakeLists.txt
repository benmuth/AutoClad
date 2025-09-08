cmake_minimum_required(VERSION 3.19)
project(
        sts_lightspeed
        VERSION 0.1
        DESCRIPTION "High Performance, rng accurate implementation of Slay the Spire with tools for tree search and ml."
        LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file (GLOB_RECURSE sts_lightspeed_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Create a filtered source list for battle executable (exclude conflicting files)
set(sts_battle_SOURCES ${sts_lightspeed_SOURCES})
list(REMOVE_ITEM sts_battle_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/combat/BattleContext.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/game/GameContext.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/combat/CardManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/game/Game.cpp"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS  "-Wno-shift-count-overflow -O3")
set(CMAKE_BUILD_TYPE Debug)

# Find LibTorch C++ API (required)
find_package(Torch REQUIRED)
message(STATUS "Found LibTorch C++ API at: ${TORCH_LIBRARIES}")

# Set C++17 standard required by LibTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch compile flags
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

add_subdirectory(json)

# target_include_directories(slaythespire PUBLIC include)
# target_include_directories(slaythespire PUBLIC json/single_include)
# target_include_directories(slaythespire PUBLIC bindings)


add_executable(main apps/main.cpp ${sts_lightspeed_SOURCES})
target_link_directories(main PRIVATE json::nlohmann_json)
target_include_directories(main PUBLIC include)
target_include_directories(main PUBLIC json/include)

add_executable(test apps/test.cpp ${sts_lightspeed_SOURCES} battle/src/NeuralNetAgent.cpp)
target_include_directories(test PUBLIC include)
target_link_directories(test PUBLIC json::nlohmann_json)
target_include_directories(test PUBLIC json/include)
target_include_directories(test PUBLIC battle/include)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 17)
set_property(TARGET test PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(small-test apps/small-test.cpp ${sts_lightspeed_SOURCES})
target_link_directories(small-test PRIVATE json::nlohmann_json)
target_include_directories(small-test PUBLIC include)
target_include_directories(small-test PUBLIC json/include)

# Create a minimal source list for battle executable (only essential files)
set(sts_battle_minimal_SOURCES
    "battle/src/Monster2.cpp"
    "battle/src/MonsterGroup2.cpp"
    "battle/src/MonsterMoveDamage2.cpp"
    "battle/src/MonsterSpecific2.cpp"
    "battle/src/Actions2.cpp"
    "battle/src/CardQueue2.cpp"
    "battle/src/CardInstance2.cpp"
    "src/game/Card.cpp"
)

add_executable(battle
    battle/src/main.cpp
    battle/src/FightDefinition.cpp
    battle/src/BattleContext2.cpp
    battle/src/GameContext2.cpp
    battle/src/CardManager2.cpp
    battle/src/Game2.cpp
    battle/src/Deck2.cpp
    battle/src/Player2.cpp
    ${sts_battle_minimal_SOURCES}
)
target_link_directories(battle PRIVATE json::nlohmann_json)
target_include_directories(battle PUBLIC battle/include)
target_include_directories(battle PUBLIC include)
target_include_directories(battle PUBLIC json/include)

# Create battle-agent source list (minimal sources needed for battle agent)
set(sts_battle_agent_SOURCES
    "battle/src/Monster2.cpp"
    "battle/src/MonsterGroup2.cpp"
    "battle/src/MonsterMoveDamage2.cpp"
    "battle/src/MonsterSpecific2.cpp"
    "battle/src/Actions2.cpp"
    "battle/src/CardQueue2.cpp"
    "battle/src/CardInstance2.cpp"
    "src/game/Card.cpp"
    "src/sim/SimHelpers.cpp"
)

# Add neural network agent sources
list(APPEND sts_battle_agent_SOURCES "battle/src/NeuralNetAgent.cpp")

add_executable(battle-agent
    apps/battle-agent.cpp
    battle/src/BattleContext2.cpp
    battle/src/GameContext2.cpp
    battle/src/CardManager2.cpp
    battle/src/Game2.cpp
    battle/src/Deck2.cpp
    battle/src/Player2.cpp
    battle/src/SimpleAgent2.cpp
    battle/src/AutoClad.cpp
    battle/src/Action2.cpp
    ${sts_battle_agent_SOURCES}
)
target_link_directories(battle-agent PRIVATE json::nlohmann_json)
target_include_directories(battle-agent PUBLIC battle/include)
target_include_directories(battle-agent PUBLIC include)
target_include_directories(battle-agent PUBLIC json/include)

# Link LibTorch C++ API
target_link_libraries(battle-agent "${TORCH_LIBRARIES}")

# Set C++17 and required compile flags for LibTorch
set_property(TARGET battle-agent PROPERTY CXX_STANDARD 17)
set_property(TARGET battle-agent PROPERTY CXX_STANDARD_REQUIRED ON)

# Copy model files to build directory for runtime access
add_custom_command(TARGET battle-agent POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/AutoClad/jaw_worm_model_traced.pt"
    "$<TARGET_FILE_DIR:battle-agent>/jaw_worm_model_traced.pt")
add_custom_command(TARGET battle-agent POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/AutoClad/scaler_params.json"
    "$<TARGET_FILE_DIR:battle-agent>/scaler_params.json")
