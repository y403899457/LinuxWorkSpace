project (mycudalib)

find_package(CUDA REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS *.cuh *.h)
file(GLOB CURRENT_SOURCES *.cu *.cpp *.cc)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(mycudalib SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
