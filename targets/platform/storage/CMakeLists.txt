plce_add_library(plce_platform_storage_stub SOURCES
    stub/StubStorage.cpp
)

target_include_directories(plce_platform_storage_stub
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

add_library(plce_platform_storage ALIAS plce_platform_storage_stub)
