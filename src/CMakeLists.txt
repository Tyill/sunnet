cmake_minimum_required(VERSION 3.0.2)

PROJECT("skynet")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Builds/$<CONFIG>/${CMAKE_VS_PLATFORM_NAME})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Builds/$<CONFIG>/${CMAKE_VS_PLATFORM_NAME})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Builds/$<CONFIG>/${CMAKE_VS_PLATFORM_NAME})

set(GPU "NONE" CACHE STRING "Selected GPU library")
set_property(CACHE GPU PROPERTY STRINGS "NONE;CUDA;cuDNN")

if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  link_directories(";$(OutDir)")
endif(MSVC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(snAux)
add_subdirectory(snBase)
add_subdirectory(snEngine)
add_subdirectory(snOperator)
add_subdirectory(skynet)
add_subdirectory(snExample)