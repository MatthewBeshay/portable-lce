set(PLCE_INPUT_SOURCES sdl2/SDL2Input.cpp)
if(APPLE)
    list(APPEND PLCE_INPUT_SOURCES sdl2/SDL2CursorPatch.m)
endif()

plce_add_library(plce_platform_input_sdl2 SOURCES ${PLCE_INPUT_SOURCES})

target_include_directories(plce_platform_input_sdl2
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(plce_platform_input_sdl2 PRIVATE SDL2::SDL2)

if(APPLE)
    target_link_libraries(plce_platform_input_sdl2 PRIVATE "-framework AppKit")
    set_source_files_properties(sdl2/SDL2CursorPatch.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc")
endif()

add_library(plce_platform_input ALIAS plce_platform_input_sdl2)
