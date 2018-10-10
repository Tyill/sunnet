cmake_minimum_required(VERSION 3.1)

project(snBase)

FILE(GLOB_RECURSE SRC_snBase "*.cpp" "*.h")

add_library(snBase INTERFACE)