plce_add_library(plce_platform_renderer_bgfx SOURCES
    bgfx/BgfxRenderPath.cpp
    bgfx/gl_stubs.cpp
    RenderPathGlobals.cpp
)

target_include_directories(plce_platform_renderer_bgfx
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
            ${Stb_INCLUDE_DIR}
)

target_link_libraries(plce_platform_renderer_bgfx
    PUBLIC  plce_java
    PRIVATE bgfx
            bimg
            bx
            SDL2::SDL2
            glm::glm
)

if(APPLE)
    set_source_files_properties(
        bgfx/BgfxRenderPath.cpp
        bgfx/gl_stubs.cpp
        RenderPathGlobals.cpp
        PROPERTIES COMPILE_FLAGS "-xobjective-c++"
    )
endif()

add_library(plce_platform_renderer ALIAS plce_platform_renderer_bgfx)
