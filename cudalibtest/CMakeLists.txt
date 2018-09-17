cmake_minimum_required (VERSION 2.8)

project (mycudalibtest)

aux_source_directory (. SRC_DIR)

add_subdirectory (mycudalib)

add_executable (cudalibtest ${SRC_DIR})

target_link_libraries (cudalibtest mycudalib)
