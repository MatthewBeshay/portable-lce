plce_add_library(plce_platform_profile_stub SOURCES
    stub/StubProfile.cpp
)

target_include_directories(plce_platform_profile_stub
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

add_library(plce_platform_profile ALIAS plce_platform_profile_stub)
