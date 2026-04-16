set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS common_sources.txt)
file(STRINGS common_sources.txt APP_COMMON_SOURCES)

set(APP_PLATFORM_SOURCES desktop/main.cpp)

add_executable(Minecraft.Client
    ${APP_COMMON_SOURCES}
    ${APP_PLATFORM_SOURCES}
)

target_include_directories(Minecraft.Client
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${Stb_INCLUDE_DIR}
            ${MINIAUDIO_INCLUDE_DIR}
)

target_link_libraries(Minecraft.Client
    PRIVATE plce_global_defs
            plce_minecraft
            plce_assets_localisation
            plce_platform_fs
            plce_platform_game
            plce_platform_input
            plce_platform_leaderboard
            plce_platform_network
            plce_platform_profile
            plce_platform_renderer
            plce_platform_sound
            plce_platform_storage
            plce_platform_thread
            SDL2::SDL2  # for iggy gdraw, remove once that path is gone
)

if(TARGET GLEW::GLEW)
    target_link_libraries(Minecraft.Client PRIVATE GLEW::GLEW)
endif()

# mimalloc must be linked first so the loader resolves malloc/free to it
# before libc. On ELF that needs --no-as-needed to defeat the linker
# dropping the dep as "unreferenced".
if(PLCE_ENABLE_MIMALLOC)
    find_package(mimalloc CONFIG REQUIRED)
    if(UNIX AND NOT APPLE)
        target_link_options(Minecraft.Client PRIVATE
            "LINKER:--no-as-needed"
            "LINKER:-lmimalloc"
            "LINKER:--as-needed"
        )
    endif()
    target_link_libraries(Minecraft.Client PRIVATE mimalloc)
endif()

if(PLCE_UI_BACKEND STREQUAL "shiggy")
    target_link_libraries(Minecraft.Client PRIVATE shiggy)
    if(WIN32)
        add_custom_command(TARGET Minecraft.Client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:shiggy>
                    $<TARGET_FILE_DIR:Minecraft.Client>
            VERBATIM
        )
    endif()
endif()

install(TARGETS Minecraft.Client RUNTIME DESTINATION .)

# Stage assets next to the exe so it runs from the build tree too.
# Multi-config generators put the exe under <dir>/<config>/.
add_custom_command(
    TARGET Minecraft.Client POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/copy_assets_to_app.py
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_BINARY_DIR}
            $<TARGET_FILE_DIR:Minecraft.Client>
            ${MEDIA_ARCHIVE_OUT}
            $<TARGET_FILE_DIR:Minecraft.Client>/assets.stamp
    COMMENT "Copying assets alongside Minecraft.Client"
    VERBATIM
)

add_dependencies(Minecraft.Client plce_media_archive)
