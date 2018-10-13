cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snOperator)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h" "*.cu")

add_library(snOperator STATIC ${SRC_snOperator})

if(USE_CUDA)

    set_target_properties(snOperator PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(snOperator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_features(snOperator PUBLIC cxx_std_11)

    add_definitions(-DSN_CUDA)

    target_link_libraries(snOperator cublas cudart_static)

    if(GPU STREQUAL "CUDNN")
     add_definitions(-DSN_CUDNN)
	 target_link_libraries(snOperator cudnn)
  endif()
endif()

target_link_libraries(snOperator snAux snBase openblas)
