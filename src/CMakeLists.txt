cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

PROJECT(skynet)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Builds/$<CONFIG>/${CMAKE_VS_PLATFORM_NAME})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Builds/$<CONFIG>/${CMAKE_VS_PLATFORM_NAME})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Builds/$<CONFIG>/${CMAKE_VS_PLATFORM_NAME})

set(GPU "NONE" CACHE STRING "Selected GPU library")
set_property(CACHE GPU PROPERTY STRINGS "NONE;CUDA;CUDNN")

option(USE_AVX     "AVX library support"    ON)

if (GPU STREQUAL "CUDA" OR GPU STREQUAL "CUDNN")
  set(USE_CUDA ON)
else()
  set(USE_CUDA OFF)
endif()

if(USE_CUDA)
  enable_language("CUDA")
  find_package(CUDA REQUIRED)
  include_directories( ${CUDA_INCLUDE_DIR})
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (UNIX)
  find_package(BLAS REQUIRED)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fPIC")

 if (USE_AVX)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
 endif()
endif()

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