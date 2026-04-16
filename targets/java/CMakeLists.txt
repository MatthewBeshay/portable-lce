plce_add_library(plce_java SOURCES
    src/InputOutputStream/BufferedOutputStream.cpp
    src/InputOutputStream/BufferedReader.cpp
    src/InputOutputStream/ByteArrayInputStream.cpp
    src/InputOutputStream/ByteArrayOutputStream.cpp
    src/InputOutputStream/DataInputStream.cpp
    src/InputOutputStream/DataOutputStream.cpp
    src/InputOutputStream/FileInputStream.cpp
    src/InputOutputStream/FileOutputStream.cpp
    src/InputOutputStream/InputStream.cpp
    src/InputOutputStream/InputStreamReader.cpp
    src/Buffer.cpp
    src/ByteBuffer.cpp
    src/Class.cpp
    src/Color.cpp
    src/File.cpp
    src/FloatBuffer.cpp
    src/IntBuffer.cpp
    src/JavaMath.cpp
    src/Random.cpp
    src/system.cpp
)

target_include_directories(plce_java
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(plce_java PUBLIC plce_util)
