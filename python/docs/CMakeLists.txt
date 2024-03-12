#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

find_package(Sphinx)

if(Sphinx_FOUND)
  message(STATUS "Sphinx found")
elseif(NURI_BUILD_PYTHON_DOCS)
  message(FATAL_ERROR "Sphinx not found; cannot build docs")
else()
  message(STATUS "Sphinx not found; not building docs")
  return()
endif()

configure_file(conf.py.in conf.py @ONLY)
add_sphinx_docs(nuri_python_docs)
