cmake_minimum_required(VERSION 3.0.2)

project(snAux)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_snAux "*.cpp" "*.h")

add_library(snAux STATIC ${SRC_snAux})