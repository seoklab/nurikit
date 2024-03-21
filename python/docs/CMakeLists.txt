#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

configure_file(conf.py.in "${CMAKE_CURRENT_LIST_DIR}/conf.py" @ONLY)

find_package(Sphinx)

if(Sphinx_FOUND)
  message(STATUS "Sphinx found")
elseif(NURI_BUILD_PYTHON_DOCS)
  message(FATAL_ERROR "Sphinx not found; cannot build docs")
else()
  message(STATUS "Sphinx not found; not building docs")
  return()
endif()

add_sphinx_docs(nuri_python_docs)
