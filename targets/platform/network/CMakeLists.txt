plce_add_library(plce_platform_network_stub SOURCES
    stub/StubNetworkPlayer.cpp
    stub/StubPlatformNetwork.cpp
)

target_include_directories(plce_platform_network_stub
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(plce_platform_network_stub PUBLIC plce_java)

add_library(plce_platform_network ALIAS plce_platform_network_stub)
