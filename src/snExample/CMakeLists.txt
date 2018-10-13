cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snExample)

FILE(GLOB_RECURSE SRC_exmpl "*.cpp" "*.h")

add_executable(snExample ${SRC_exmpl})

target_link_libraries(snExample skynet)