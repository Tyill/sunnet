cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snOperatorCPU)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h" "*.cu")

add_library(snOperatorCPU STATIC ${SRC_snOperator})

if(USE_AVX)
   add_definitions(-DSN_AVX)
         
   target_link_libraries(snOperatorCPU snSIMD)
endif()

target_link_libraries(snOperatorCPU snAux snBase)

if(WIN32)
  target_link_libraries(snOperatorCPU libopenblas)
else()
  target_link_libraries(snOperatorCPU openblas)
endif()