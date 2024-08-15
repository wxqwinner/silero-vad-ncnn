cmake_minimum_required(VERSION 3.10)

project(silero-vad-ncnn)

set(CMAKE_CXX_STANDARD 11)

# ncnn library
include_directories(${CMAKE_SOURCE_DIR}/../3rdParty/ncnn/include)
link_directories(${CMAKE_SOURCE_DIR}/../3rdParty/ncnn/lib)

# src files
set(SRC_LISTS
    ncnn_vad.cpp
    vad_example.cpp
)

# add executable
add_executable(silero-vad-ncnn ${SRC_LISTS})
target_link_libraries(silero-vad-ncnn PRIVATE ncnn)
