find_package(Doxygen 1.8)

if(DOXYGEN_FOUND)
  message(STATUS "Doxygen found")
else()
  if(NURIKIT_BUILD_DOCS)
    message(WARNING "Doxygen not found; cannot build docs")
  else()
    message(STATUS "Doxygen not found; not building docs")
  endif()
  return()
endif()

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
add_custom_target(nurikit_docs
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM)

if(NURIKIT_BUILD_DOCS)
  set_target_properties(nurikit_docs PROPERTIES EXCLUDE_FROM_ALL OFF)
endif()
