cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snOperator)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h" "*.cu")

add_library(snOperator STATIC ${SRC_snOperator})

set_target_properties(snOperator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(snOperator PUBLIC cxx_std_11)

target_link_libraries(snOperator cublas cudart_static cudnn snAux snBase)