cmake_minimum_required (VERSION 2.8)

project (cudastudy)

add_subdirectory(GpuReduce)

aux_source_directory(. SRC_DIR)

add_executable(cudastudy ${SRC_DIR})

target_link_libraries(cudastudy cudareducelib)
