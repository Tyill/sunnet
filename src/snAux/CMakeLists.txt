cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snAux)

FILE(GLOB_RECURSE SRC_snAux "*.cpp" "*.h")

add_library(snAux STATIC ${SRC_snAux})