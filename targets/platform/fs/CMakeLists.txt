plce_add_library(plce_platform_fs_std SOURCES
    std/StdFilesystem.cpp
)

target_include_directories(plce_platform_fs_std
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

add_library(plce_platform_fs ALIAS plce_platform_fs_std)
