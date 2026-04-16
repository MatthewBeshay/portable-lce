plce_add_library(plce_platform_game_stub SOURCES
    stub/StubPlatformGame.cpp
)

target_include_directories(plce_platform_game_stub
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

add_library(plce_platform_game ALIAS plce_platform_game_stub)
