cmake_minimum_required(VERSION 3.28)

project(portable-lce
    VERSION 0.1.0
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_UNITY_BUILD_BATCH_SIZE 8)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT PLCE_IPO_SUPPORTED OUTPUT PLCE_IPO_ERROR)
if(PLCE_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
endif()

find_program(PLCE_COMPILER_LAUNCHER NAMES sccache ccache)
if(PLCE_COMPILER_LAUNCHER)
    set(CMAKE_C_COMPILER_LAUNCHER   ${PLCE_COMPILER_LAUNCHER})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${PLCE_COMPILER_LAUNCHER})
endif()

# ----- Project options --------------------------------------------------------

set(PLCE_RENDERER "bgfx" CACHE STRING "Rendering backend (bgfx | vulkan)")
set_property(CACHE PLCE_RENDERER PROPERTY STRINGS "bgfx" "vulkan")

set(PLCE_UI_BACKEND "shiggy" CACHE STRING "UI backend (shiggy | java)")
set_property(CACHE PLCE_UI_BACKEND PROPERTY STRINGS "shiggy" "java")

option(PLCE_CLASSIC_PANORAMA
    "Enable classic Java Edition panorama (ui_backend=java only)" OFF)

option(PLCE_ENABLE_VSYNC "Enable V-Sync by default" ON)

option(PLCE_ENABLE_FRAME_PROFILER
    "Enable the in-engine frame profiler for render hotspot discovery" OFF)

set(PLCE_OCCLUSION_CULLING "frustum" CACHE STRING
    "Occlusion culling mode (off | frustum | bfs | hardware)")
set_property(CACHE PLCE_OCCLUSION_CULLING PROPERTY STRINGS
    "off" "frustum" "bfs" "hardware")

option(PLCE_ENABLE_MIMALLOC
    "Link mimalloc as the system allocator (requires vcpkg 'mimalloc' feature)" OFF)

set(PLCE_ISA_BASELINE "x86-64-v3" CACHE STRING
    "Baseline ISA for non-MSVC compilers (off | native | x86-64-v2 | x86-64-v3 | x86-64-v4)")
set_property(CACHE PLCE_ISA_BASELINE PROPERTY STRINGS
    "off" "native" "x86-64-v2" "x86-64-v3" "x86-64-v4")

option(PLCE_ENABLE_HARDENING "Enable stack/CFI/fortify hardening flags" ON)
option(PLCE_ENABLE_WARNINGS  "Enable -Wall/-Wextra-equivalent warnings" ON)

# ----- Global compile defs / flags --------------------------------------------

add_library(plce_global_defs INTERFACE)

target_compile_definitions(plce_global_defs INTERFACE
    SPLIT_SAVES
    _LARGE_WORLDS
    _EXTENDED_ACHIEVEMENTS
    MULTITHREAD_ENABLE
    $<$<CONFIG:Debug,RelWithDebInfo>:_DEBUG_MENUS_ENABLED>
)

if(PLCE_ENABLE_VSYNC)
    target_compile_definitions(plce_global_defs INTERFACE ENABLE_VSYNC)
endif()
if(PLCE_CLASSIC_PANORAMA)
    target_compile_definitions(plce_global_defs INTERFACE CLASSIC_PANORAMA)
endif()
if(PLCE_ENABLE_FRAME_PROFILER)
    target_compile_definitions(plce_global_defs INTERFACE ENABLE_FRAME_PROFILER)
endif()

if(PLCE_UI_BACKEND STREQUAL "shiggy")
    target_compile_definitions(plce_global_defs INTERFACE _ENABLEIGGY)
elseif(PLCE_UI_BACKEND STREQUAL "java")
    target_compile_definitions(plce_global_defs INTERFACE ENABLE_JAVA_GUIS)
endif()

if(PLCE_OCCLUSION_CULLING STREQUAL "off")
    target_compile_definitions(plce_global_defs INTERFACE OCCLUSION_MODE_NONE)
elseif(PLCE_OCCLUSION_CULLING STREQUAL "frustum")
    target_compile_definitions(plce_global_defs INTERFACE OCCLUSION_MODE_FRUSTUM)
elseif(PLCE_OCCLUSION_CULLING STREQUAL "bfs")
    target_compile_definitions(plce_global_defs INTERFACE
        OCCLUSION_MODE_BFS USE_OCCLUSION_CULLING)
elseif(PLCE_OCCLUSION_CULLING STREQUAL "hardware")
    target_compile_definitions(plce_global_defs INTERFACE
        OCCLUSION_MODE_HARDWARE USE_OCCLUSION_CULLING)
endif()

if(PLCE_ENABLE_WARNINGS)
    target_compile_options(plce_global_defs INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /utf-8>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor>
    )
endif()

# Keep frame pointers so sampling profilers produce sane stacks.
target_compile_options(plce_global_defs INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/Oy->
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-omit-frame-pointer>
)

if(PLCE_ENABLE_HARDENING)
    # /guard:ehcont is intentionally omitted: it requires every linked
    # object to participate, which third-party deps fetched via
    # FetchContent (bgfx, bx, bimg) do not.
    target_compile_options(plce_global_defs INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/GS /guard:cf>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fstack-protector-strong>
        # _FORTIFY_SOURCE needs an optimised build to be effective.
        $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<NOT:$<CONFIG:Debug>>>:-D_FORTIFY_SOURCE=2>
        # CET on x86; safe NOP on chips without CET hardware.
        $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<NOT:$<PLATFORM_ID:Darwin>>>:-fcf-protection=full>
    )
    target_link_options(plce_global_defs INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/guard:cf /DYNAMICBASE /NXCOMPAT /HIGHENTROPYVA>
    )
endif()

if(NOT PLCE_ISA_BASELINE STREQUAL "off")
    target_compile_options(plce_global_defs INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=${PLCE_ISA_BASELINE}>
    )
endif()

# ----- vcpkg-managed dependencies --------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(simdutf CONFIG REQUIRED)
find_package(Stb REQUIRED)
# miniaudio: header-only in vcpkg, no CMake config; consumers do
# find_path(MINIAUDIO_INCLUDE_DIR "miniaudio.h").
find_path(MINIAUDIO_INCLUDE_DIR "miniaudio.h" REQUIRED)

# GLEW remains while a few legacy translation units still pull it in.
find_package(GLEW CONFIG QUIET)

if(PLCE_RENDERER STREQUAL "vulkan")
    find_package(Vulkan REQUIRED)
    find_package(volk CONFIG REQUIRED)
    find_package(VulkanMemoryAllocator CONFIG REQUIRED)
endif()

# ----- bgfx via FetchContent --------------------------------------------------

if(PLCE_RENDERER STREQUAL "bgfx")
    include(FetchContent)
    FetchContent_Declare(
        bgfx_cmake
        GIT_REPOSITORY https://github.com/bkaradzic/bgfx.cmake.git
        GIT_TAG        v1.143.9216-529
        GIT_SHALLOW    TRUE
        GIT_SUBMODULES_RECURSE TRUE
    )
    set(BGFX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BGFX_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(BGFX_INSTALL        OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(bgfx_cmake)
endif()

# ----- shiggy via FetchContent -----------------------------------------------
#
# Private repo shipping a prebuilt iggy library per platform. The Linux
# path requires extracting objects from a PS4 static archive; that
# pipeline is not yet translated from the meson build.

if(PLCE_UI_BACKEND STREQUAL "shiggy")
    include(FetchContent)
    FetchContent_Declare(
        shiggy
        GIT_REPOSITORY https://github.com/portable-lce/shiggy.git
        GIT_TAG        8d3f8dea3a6baefc24d071e0d98b041de3881e1a
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(shiggy)

    if(WIN32)
        add_library(shiggy SHARED IMPORTED GLOBAL)
        set_target_properties(shiggy PROPERTIES
            IMPORTED_LOCATION ${shiggy_SOURCE_DIR}/libs/iggy_w64.dll
            IMPORTED_IMPLIB   ${shiggy_SOURCE_DIR}/libs/iggy_w64.lib
            INTERFACE_INCLUDE_DIRECTORIES ${shiggy_SOURCE_DIR}/include
        )
    else()
        message(FATAL_ERROR
            "shiggy on non-Windows platforms requires the PS4 "
            "libiggy_orbis.a object-extraction pipeline, which is not "
            "yet translated. Build with -DPLCE_UI_BACKEND=java.")
    endif()
endif()

# ----- Helpers ---------------------------------------------------------------

include(cmake/PlceHelpers.cmake)

# ----- Subdirectories --------------------------------------------------------

add_subdirectory(targets/util)
add_subdirectory(targets/java)
add_subdirectory(targets/nbt)
add_subdirectory(targets/platform)
add_subdirectory(targets/resources)
add_subdirectory(targets/minecraft)
add_subdirectory(targets/app)
