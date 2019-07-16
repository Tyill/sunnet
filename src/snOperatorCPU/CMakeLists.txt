cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(snOperator)

FILE(GLOB_RECURSE SRC_snOperator "*.cpp" "*.h" "*.cu")

add_library(snOperator STATIC ${SRC_snOperator})

if(USE_AVX)
   add_definitions(-DSN_AVX)
         
   target_link_libraries(snOperator snSIMD)
endif()

target_link_libraries(snOperator snAux snBase)

if(WIN32)
  target_link_libraries(snOperator libopenblas)
else()
  target_link_libraries(snOperator openblas)
endif()