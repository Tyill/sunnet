cmake_minimum_required(VERSION 3.1)

project(snOperator)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h" "*.cu")

add_library(snOperator STATIC ${SRC_snOperator})

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(GPU STREQUAL "CUDA" OR GPU STREQUAL "cuDNN")
  find_package(CUDA REQUIRED)
  include_directories( ${CUDA_INCLUDE_DIR})
  add_definitions(-DSN_CUDA)
 
  link_directories(${CUDA_INCLUDE_DIRS}/../lib/x64)
    
  target_link_libraries(snOperator cublas cudart_static)
  
  if(GPU STREQUAL "cuDNN")
     add_definitions(-DSN_CUDNN)
	 target_link_libraries(snOperator cudnn)
  endif()	 
endif()

target_link_libraries(snOperator snAux snBase libopenblas)