cmake_minimum_required(VERSION 3.1)

project(skynet)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_skynet "*.cpp" "*.h")

if(MSVC)
  add_definitions(-DSKYNETDLL_EXPORTS)
  set_target_properties(skynet PROPERTIES OUTPUT_NAME "libmxnet")
endif(MSVC)

add_library(skynet SHARED ${SRC_skynet})

target_link_libraries(skynet snAux snBase snEngine snOperator)

# copy mswin dependent
if(WIN32)
   mswin_getDependDir(DEPEND_WIN_DIR)
   add_custom_command(TARGET skynet PRE_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DEPEND_WIN_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()