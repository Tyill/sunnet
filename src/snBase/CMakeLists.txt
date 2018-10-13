cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snBase)

FILE(GLOB_RECURSE SRC_snBase "*.cpp" "*.h")

add_library(snBase INTERFACE)