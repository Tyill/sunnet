cmake_minimum_required(VERSION 3.1)

project(libskynet)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_skynet "*.cpp" "*.h")

if(MSVC)
  add_definitions(-DSKYNETDLL_EXPORTS)
endif(MSVC)

if (UNIX)
   add_compile_options(-fPIC)
endif(UNIX)

add_library(libskynet SHARED ${SRC_skynet})

target_link_libraries(libskynet snAux snBase snEngine snOperator)

# copy mswin dependent
if(WIN32)
   mswin_getDependDir(DEPEND_WIN_DIR)
   add_custom_command(TARGET libskynet PRE_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DEPEND_WIN_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()