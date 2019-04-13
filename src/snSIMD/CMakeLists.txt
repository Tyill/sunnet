cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snEngine)

FILE(GLOB_RECURSE SRC_snEngine "*.cpp" "*.h")

add_library(snEngine STATIC ${SRC_snEngine})

target_link_libraries(snEngine snAux snBase)