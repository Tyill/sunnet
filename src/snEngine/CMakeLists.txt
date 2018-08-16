cmake_minimum_required(VERSION 3.0.2)

project(snEngine)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_snEngine "*.cpp" "*.h")

add_library(snEngine STATIC ${SRC_snEngine})

target_link_libraries(snEngine snAux snBase)