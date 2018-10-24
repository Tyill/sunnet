cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(skynet)

FILE(GLOB_RECURSE SRC_skynet "*.cpp" "*.h")

add_library(skynet SHARED ${SRC_skynet})

if(MSVC)
  add_definitions(-DSKYNETDLL_EXPORTS)
  set_target_properties(skynet PROPERTIES OUTPUT_NAME "libskynet")
endif(MSVC)

# copy mswin dependent
if(WIN32)
  include(../cmake/mswin.cmake)
  mswin_getDependDir(DEPEND_WIN_DIR)
  add_custom_command(TARGET skynet PRE_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory ${DEPEND_WIN_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

target_link_libraries(skynet snAux snBase snEngine snOperator)