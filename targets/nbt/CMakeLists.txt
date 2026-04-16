plce_add_library(plce_nbt SOURCES
    src/NbtIo.cpp
    src/Tag.cpp
)

target_include_directories(plce_nbt
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(plce_nbt PUBLIC plce_java)
