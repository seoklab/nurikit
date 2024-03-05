#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#
include(FetchContent)

Fetchcontent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.4
)
nuri_make_available_deponly(pybind11)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/nuri/_version.py.in"
  "${CMAKE_CURRENT_LIST_DIR}/nuri/_version.py"
  @ONLY)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/nuri/_version.py"
  DESTINATION ".")

include_directories("${CMAKE_CURRENT_LIST_DIR}/nuri")
file(GLOB_RECURSE NURI_PYTHON_SOURCES "nuri/*.cpp")

add_custom_target(nuri_python ALL
  SOURCES ${NURI_PYTHON_SOURCES})
add_dependencies(nuri_python nuri_lib)

foreach(NURI_PYTHON_SOURCE IN LISTS NURI_PYTHON_SOURCES)
  get_filename_component(
    NURI_PYTHON_SOURCE_DIR "${NURI_PYTHON_SOURCE}" DIRECTORY)
  set(NURI_PYTHON_MODULE_OUTPUT_DIR "${NURI_PYTHON_SOURCE_DIR}")
  file(RELATIVE_PATH NURI_PYTHON_SOURCE_DIR_INV
    "${NURI_PYTHON_SOURCE_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/nuri")
  file(RELATIVE_PATH NURI_PYTHON_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/nuri"
    "${NURI_PYTHON_SOURCE_DIR}")
  get_filename_component(
    NURI_PYTHON_MODULE_NAME "${NURI_PYTHON_SOURCE}" NAME_WE)

  file(RELATIVE_PATH NURI_PYTHON_MODULE
    "${CMAKE_CURRENT_LIST_DIR}/nuri"
    "${NURI_PYTHON_SOURCE}")
  file(TO_CMAKE_PATH "${NURI_PYTHON_MODULE}" NURI_PYTHON_MODULE)
  string(REPLACE "/" "_" NURI_PYTHON_MODULE "${NURI_PYTHON_MODULE}")
  string(REGEX REPLACE
    "\.cpp$" ""
    NURI_PYTHON_MODULE "${NURI_PYTHON_MODULE}")
  string(PREPEND NURI_PYTHON_MODULE "nuri_python_")

  pybind11_add_module(
    "${NURI_PYTHON_MODULE}"
    OPT_SIZE
    "${NURI_PYTHON_SOURCE}")
  target_link_libraries("${NURI_PYTHON_MODULE}"
    PRIVATE nuri_lib absl::strings)
  set_target_properties("${NURI_PYTHON_MODULE}"
    PROPERTIES
    OUTPUT_NAME "${NURI_PYTHON_MODULE_NAME}"
    INSTALL_RPATH "$ORIGIN/${NURI_PYTHON_SOURCE_DIR_INV}${CMAKE_INSTALL_LIBDIR}")

  if(NOT SKBUILD)
    set_target_properties("${NURI_PYTHON_MODULE}"
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${NURI_PYTHON_MODULE_OUTPUT_DIR}")
  endif()

  install(TARGETS "${NURI_PYTHON_MODULE}"
    LIBRARY
    DESTINATION "./${NURI_PYTHON_SOURCE_DIR}")

  add_dependencies(nuri_python "${NURI_PYTHON_MODULE}")
endforeach()

target_link_libraries(nuri_python__log_adapter PRIVATE absl::log_initialize)

add_subdirectory(docs EXCLUDE_FROM_ALL)
