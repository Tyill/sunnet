cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snOperatorCUDA)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h" "*.cu")

add_library(snOperatorCUDA STATIC ${SRC_snOperator})

set_target_properties(snOperatorCUDA PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(snOperatorCUDA PUBLIC cxx_std_11)

target_link_libraries(snOperatorCUDA cublas cudart_static cudnn snAux snBase)