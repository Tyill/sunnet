cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(skynet)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_skynet "*.cpp" "*.h")

if(MSVC)
  add_definitions(-DSKYNETDLL_EXPORTS)
  set_target_properties(skynet PROPERTIES OUTPUT_NAME "libskynet")
endif(MSVC)

add_library(skynet SHARED ${SRC_skynet})

target_link_libraries(skynet snAux snBase snEngine snOperator)