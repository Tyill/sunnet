cmake_minimum_required(VERSION 3.0.2)

project(skynet)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_skynet "*.cpp" "*.h")

if (UNIX)
   add_compile_options(-fPIC)
endif(UNIX)

add_library(skynet SHARED ${SRC_skynet})

target_link_libraries(skynet snAux snBase snEngine snOperator)