# plce_platform_renderer_ui — backend-agnostic 2D UI draw helpers.
#
# Converts typed primitives (fill, gradient, textured quad, glyph,
# fullscreen overlay, vignette) into rp::DrawCall records pushed onto
# rp::ui_overlay. Callers never see Vulkan/bgfx vocabulary — all the
# DrawCall assembly, material-handle lookup, and live-matrix snapshot
# lives here so it stops leaking into legacy client files.
#
# No dependency on the backend target at link time: symbols referenced
# from the global RenderPath (create_material, MatrixGet,
# alloc_transient_vertices) resolve through the final executable's link
# of plce_platform_renderer. This library just needs the headers + glm.

plce_add_library(plce_platform_renderer_ui SOURCES
    UiDraw.cpp
)

target_include_directories(plce_platform_renderer_ui
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

target_link_libraries(plce_platform_renderer_ui
    PRIVATE glm::glm
)
