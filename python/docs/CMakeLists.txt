#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

option(NURIKIT_PYDOC_DEPEND_PYTHON
  "Auto-rebuild python module before building docs" OFF)
mark_as_advanced(NURIKIT_PYDOC_DEPEND_PYTHON)

find_package(Sphinx)

if(Sphinx_FOUND)
  message(STATUS "Sphinx found")
else()
  if(NURIKIT_BUILD_DOCS)
    message(WARNING "Sphinx not found; cannot build docs")
  else()
    message(STATUS "Sphinx not found; not building docs")
  endif()
  return()
endif()

configure_file(conf.py.in conf.py @ONLY)
add_sphinx_docs(nurikit_python_docs)

if(NURIKIT_PYDOC_DEPEND_PYTHON)
  message("nurikit_python target will rebuild python bindings as requested")
  add_dependencies(nurikit_python_docs nuri_python)
endif()

if(NURIKIT_BUILD_DOCS)
  set_target_properties(nurikit_python_docs PROPERTIES EXCLUDE_FROM_ALL OFF)
endif()
