cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snSIMD)

FILE(GLOB_RECURSE SRC_snSIMD "*.cpp" "*.h")

add_library(snSIMD STATIC ${SRC_snSIMD})

if(WIN32)
   target_compile_options(snSIMD PRIVATE /arch:AVX)
endif()

target_link_libraries(snSIMD snBase)