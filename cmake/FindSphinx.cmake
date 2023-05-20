find_program(SPHINX_EXECUTABLE
  NAMES sphinx-build)
mark_as_advanced(SPHINX_EXECUTABLE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)

function(add_sphinx_docs target)
  add_custom_target("${target}"
    COMMAND ${SPHINX_EXECUTABLE}
    -E
    -b html
    -c ${CMAKE_CURRENT_BINARY_DIR}
    -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Building Sphinx documentation for ${target}"
    VERBATIM)

  add_custom_target("${target}_doctest"
    COMMAND ${SPHINX_EXECUTABLE}
    -E
    -b doctest
    -c ${CMAKE_CURRENT_BINARY_DIR}
    -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Running doctest for ${target}"
    VERBATIM)
  add_dependencies("${target}_doctest" "${target}")
endfunction()
