# Backend-agnostic 2D UI helpers. Builds regardless of the selected
# backend — the module's impl only needs rp:: headers + glm; symbols
# from the active backend (create_material, MatrixGet, ...) resolve
# when the final executable links plce_platform_renderer.
add_subdirectory(ui)

# Backend-agnostic 3D world-draw helpers (MeshBuilder for entity /
# tile-entity / particle / sky). Same shape as the ui/ module.
add_subdirectory(world)

if(PLCE_RENDERER STREQUAL "vulkan")
    find_path(VMA_INCLUDE_DIR "vma/vk_mem_alloc.h" REQUIRED)

    plce_add_library(plce_platform_renderer_vk SOURCES
        vk/Device.cpp
        vk/FrameContext.cpp
        vk/PipelineCache.cpp
        vk/Swapchain.cpp
        vk/TextureManager.cpp
        vk/Renderer.cpp
        vk/vma_impl.cpp
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
        vk/shaders/basic_compact.vert kBasicCompactVertSpv)

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
