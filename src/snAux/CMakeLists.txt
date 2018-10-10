cmake_minimum_required(VERSION 3.1)

project(snAux)

FILE(GLOB_RECURSE SRC_snAux "*.cpp" "*.h")

add_library(snAux STATIC ${SRC_snAux})