plce_add_library(plce_platform_leaderboard_stub SOURCES
    stub/StubLeaderboard.cpp
)

target_include_directories(plce_platform_leaderboard_stub
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

add_library(plce_platform_leaderboard ALIAS plce_platform_leaderboard_stub)
