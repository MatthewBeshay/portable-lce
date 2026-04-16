if(PLCE_RENDERER STREQUAL "vulkan")
    find_path(VMA_INCLUDE_DIR "vma/vk_mem_alloc.h" REQUIRED)

    plce_add_library(plce_platform_renderer_vk SOURCES
        vk/VulkanRenderPath.cpp
        vk/render/ChunkArena.cpp
        vk/render/ChunkMetadata.cpp
        vk/render/IndirectDrawBuffer.cpp
        vk/render/StagingRing.cpp
        vk/render/TerrainRenderer.cpp
        RenderPathGlobals.cpp
        gl_stubs.cpp
    )

    target_include_directories(plce_platform_renderer_vk
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
                ${CMAKE_CURRENT_SOURCE_DIR}/vk
                ${Stb_INCLUDE_DIR}
                ${VMA_INCLUDE_DIR}
    )

    target_link_libraries(plce_platform_renderer_vk
        PUBLIC  plce_java
        PRIVATE Vulkan::Headers
                Vulkan::Loader
                SDL2::SDL2
                glm::glm
    )

    plce_compile_shader(plce_platform_renderer_vk
        vk/shaders/basic.vert kBasicVertSpv)
    plce_compile_shader(plce_platform_renderer_vk
        vk/shaders/basic.frag kBasicFragSpv)
    plce_compile_shader(plce_platform_renderer_vk
        vk/shaders/terrain_cull.comp kTerrainCullSpv)
    plce_compile_shader(plce_platform_renderer_vk
        vk/shaders/terrain.vert kTerrainVertSpv)
    plce_compile_shader(plce_platform_renderer_vk
        vk/shaders/terrain.frag kTerrainFragSpv)

    add_library(plce_platform_renderer ALIAS plce_platform_renderer_vk)
    return()
endif()

plce_add_library(plce_platform_renderer_bgfx SOURCES
    bgfx/BgfxRenderPath.cpp
    gl_stubs.cpp
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
        gl_stubs.cpp
        RenderPathGlobals.cpp
        PROPERTIES COMPILE_FLAGS "-xobjective-c++"
    )
endif()

add_library(plce_platform_renderer ALIAS plce_platform_renderer_bgfx)
