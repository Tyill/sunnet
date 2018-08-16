cmake_minimum_required(VERSION 3.0.2)

project(snOperator)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h")

add_library(snOperator STATIC ${SRC_snOperator})

target_link_libraries(snOperator snAux snBase)