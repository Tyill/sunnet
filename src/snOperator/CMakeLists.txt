cmake_minimum_required(VERSION 3.1)

project(snOperator)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h")

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (UNIX)
find_package(BLAS REQUIRED)
endif()

add_library(snOperator STATIC ${SRC_snOperator})

if(GPU STREQUAL "CUDA" OR GPU STREQUAL "cuDNN")

  add_definitions(-DSN_CUDA)

  target_link_libraries(snOperator cublas cudart_static)
  target_link_libraries(snOperator ${CMAKE_SOURCE_DIR}/cmake-build-release/Builds/Release/convolutionCUDA.o
          ${CMAKE_SOURCE_DIR}/cmake-build-release/Builds/Release/deconvolutionCUDA.o
          ${CMAKE_SOURCE_DIR}/cmake-build-release/Builds/Release/fullyConnectedCUDA.o
          ${CMAKE_SOURCE_DIR}/cmake-build-release/Builds/Release/poolingCUDA.o)

  if(GPU STREQUAL "cuDNN")
     add_definitions(-DSN_CUDNN)
	 target_link_libraries(snOperator cudnn)
  endif()
endif()

target_link_libraries(snOperator snAux snBase openblas)