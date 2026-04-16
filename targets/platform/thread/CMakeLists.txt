plce_add_library(plce_platform_thread SOURCES
    C4JThread.cpp
    ShutdownManager.cpp
)

target_include_directories(plce_platform_thread
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(plce_platform_thread PUBLIC Threads::Threads)
