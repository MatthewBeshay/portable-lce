plce_add_library(plce_platform_sound_miniaudio SOURCES
    miniaudio/MiniaudioSound.cpp
)

target_include_directories(plce_platform_sound_miniaudio
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
            ${MINIAUDIO_INCLUDE_DIR}
)

add_library(plce_platform_sound ALIAS plce_platform_sound_miniaudio)
