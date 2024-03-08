if(NOT CMAKE_SCRIPT_MODE_FILE)
  message(WARNING "This file must be executed with CMake in script mode, not included with include() or add_subdirectory().")
  return()
endif()

if(NOT CMAKE_ARGV1)
  message(FATAL_ERROR "No parent directory provided.")
endif()

file(GLOB_RECURSE nuri_stubs LIST_DIRECTORIES OFF "*.pyi")

if(nuri_stubs)
  list(JOIN nuri_stubs ", \n" nuri_stubs_msg)
  message(NOTICE "Removing the following stubs:\n${nuri_stubs_msg}")
  file(REMOVE ${nuri_stubs})
endif()
