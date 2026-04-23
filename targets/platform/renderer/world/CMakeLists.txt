# plce_platform_renderer_world — backend-agnostic 3D world-draw helpers.
#
# MeshBuilder accumulates quad geometry from entity / tile-entity /
# particle / sky / weather code and emits rp::DrawCall records onto
# the rp::world_draws buckets that feed ViewDesc's world_* spans.
# Sibling of plce_platform_renderer_ui — same shape (create_material
# lookups at init(), live matrix snapshot, no Vulkan/bgfx vocabulary
# in the public header).

plce_add_library(plce_platform_renderer_world SOURCES
    WorldDraw.cpp
)

target_include_directories(plce_platform_renderer_world
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

target_link_libraries(plce_platform_renderer_world
    PRIVATE glm::glm
)
