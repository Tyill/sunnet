cmake_minimum_required(VERSION 3.0.2)

project(SVAuxFunc)

include(../cmake/mswin.cmake)

FILE(GLOB_RECURSE SRC_SVAux "*.cpp" "*.h")

if (UNIX)
   add_compile_options(-fPIC)
endif(UNIX)

add_library(SVAuxFunc STATIC ${SRC_SVAux})

if(WIN32)
   target_link_libraries(SVAuxFunc libuv ws2_32 Iphlpapi psapi Userenv Mswsock AdvApi32)
else()
   target_link_libraries(SVAuxFunc uv)
endif()


# copy mswin dependent
if(WIN32)
   mswin_getDependDir(DEPEND_WIN_DIR)
   add_custom_command(TARGET SVAuxFunc PRE_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DEPEND_WIN_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

endif()