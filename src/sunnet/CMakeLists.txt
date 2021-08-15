cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(sunnet)

FILE(GLOB_RECURSE SRC_sunnet "*.cpp" "*.h")

add_library(sunnet SHARED ${SRC_sunnet})

if(MSVC)
  add_definitions(-DSUNNETDLL_EXPORTS)
  set_target_properties(sunnet PROPERTIES OUTPUT_NAME "libsunnet")
endif(MSVC)

# copy mswin dependent
if(WIN32)
  include(../cmake/mswin.cmake)
  mswin_getDependDir(DEPEND_WIN_DIR)
  add_custom_command(TARGET sunnet PRE_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory ${DEPEND_WIN_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

target_link_libraries(sunnet snAux snBase snEngine)

if(USE_CUDA)
 target_link_libraries(sunnet snOperatorCUDA)
else()
 target_link_libraries(sunnet snOperatorCPU)
endif()