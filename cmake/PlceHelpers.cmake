# PlceHelpers.cmake — convenience wrappers for portable-lce targets.

# plce_add_library(<name> SOURCES src1.cpp src2.cpp ...)
#
# Creates a static library with unity build enabled and links the global
# compile-definitions target.
function(plce_add_library TARGET_NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    add_library(${TARGET_NAME} STATIC ${ARG_SOURCES})
    set_target_properties(${TARGET_NAME} PROPERTIES UNITY_BUILD ON)
    target_link_libraries(${TARGET_NAME} PUBLIC plce_global_defs)
endfunction()

# plce_compile_shader(<target> <glsl_source_relative> <variable_name>)
#
# Compiles a GLSL file to SPIR-V at build time using glslangValidator,
# then embeds the binary as a C header (static const uint32_t <var>[]).
# The generated header is placed next to the source under the build tree
# and the target's include path is extended so shaders can
#     #include "vk/shaders/<file>.spv.h"
function(plce_compile_shader TARGET_NAME GLSL_REL VAR_NAME)
    set(GLSL_ABS "${CMAKE_CURRENT_SOURCE_DIR}/${GLSL_REL}")
    get_filename_component(GLSL_NAME "${GLSL_REL}" NAME)
    set(SPV_H "${CMAKE_CURRENT_BINARY_DIR}/${GLSL_REL}.spv.h")

    find_program(GLSLANG_VALIDATOR glslangValidator HINTS
        "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR "glslangValidator not found — install the Vulkan SDK")
    endif()

    get_filename_component(SPV_DIR "${SPV_H}" DIRECTORY)
    file(MAKE_DIRECTORY "${SPV_DIR}")

    # Let shaders #include relative to their own directory (requires
    # GL_GOOGLE_include_directive in the shader). Any .glsl helper in
    # the same directory becomes an implicit rebuild dependency so that
    # editing common.glsl forces every dependent shader to recompile.
    get_filename_component(GLSL_DIR "${GLSL_ABS}" DIRECTORY)
    file(GLOB GLSL_INCLUDES CONFIGURE_DEPENDS "${GLSL_DIR}/*.glsl")

    add_custom_command(
        OUTPUT  "${SPV_H}"
        COMMAND "${GLSLANG_VALIDATOR}" -V
                "-I${GLSL_DIR}"
                --vn "${VAR_NAME}"
                -o "${SPV_H}"
                "${GLSL_ABS}"
        DEPENDS "${GLSL_ABS}" ${GLSL_INCLUDES}
        COMMENT "SPIR-V: ${GLSL_REL} -> ${VAR_NAME}"
        VERBATIM
    )
    target_sources(${TARGET_NAME} PRIVATE "${SPV_H}")
    target_include_directories(${TARGET_NAME}
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()
