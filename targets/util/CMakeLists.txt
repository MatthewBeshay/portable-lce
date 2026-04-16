plce_add_library(plce_util SOURCES
    StringHelpers.cpp
    FrameProfiler.cpp
)

target_include_directories(plce_util
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(plce_util PRIVATE simdutf::simdutf)
