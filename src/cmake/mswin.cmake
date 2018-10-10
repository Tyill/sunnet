function(mswin_getDependDir __string)
   set(arch "x86")
   if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(arch "x64")
   endif()
         
   get_filename_component(parentDir ${CMAKE_SOURCE_DIR} DIRECTORY)
      
   set(${__string} "${parentDir}/mswin/Dependences/${arch}" PARENT_SCOPE)
      
endfunction()