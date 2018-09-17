cmake_minimum_required (VERSION 2.8)

project(demo4)

#check if system function pow supported
include (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
check_function_exists(pow HAVE_POW)

configure_file(
"${PROJECT_SOURCE_DIR}/config.h.in"
"${PROJECT_BINARY_DIR}/config.h"
)

option(USE_MYMATH
	"Use mymath lib" ON)

if(USE_MYMATH)
	include_directories("${PROJECT_SOURCE_DIR}/MyMath")
	add_subdirectory(MyMath)
	set(EXTRA_LIBS ${EXTRA_LIBS} MyMath)
endif(USE_MYMATH)

aux_source_directory(. DIR_SRCS)

add_executable(Demo ${DIR_SRCS})
target_link_libraries(Demo ${EXTRA_LIBS})
