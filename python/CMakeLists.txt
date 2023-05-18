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
FetchContent_MakeAvailable(pybind11)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/nuri")
file(GLOB_RECURSE NURIKIT_PYTHON_SOURCES "nuri/*.cpp")

add_custom_target(nuri_python ALL
  SOURCES ${NURIKIT_PYTHON_SOURCES})
add_dependencies(nuri_python nuri_lib)

foreach(NURIKIT_PYTHON_SOURCE IN LISTS NURIKIT_PYTHON_SOURCES)
  get_filename_component(
    NURIKIT_PYTHON_SOURCE_DIR "${NURIKIT_PYTHON_SOURCE}" DIRECTORY)
  file(RELATIVE_PATH NURIKIT_PYTHON_SOURCE_DIR_INV
    "${NURIKIT_PYTHON_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/nuri")
  file(RELATIVE_PATH NURIKIT_PYTHON_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/nuri"
    "${NURIKIT_PYTHON_SOURCE_DIR}")
  get_filename_component(
    NURIKIT_PYTHON_MODULE_NAME "${NURIKIT_PYTHON_SOURCE}" NAME_WE)

  file(RELATIVE_PATH NURIKIT_PYTHON_MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/nuri"
    "${NURIKIT_PYTHON_SOURCE}")
  file(TO_CMAKE_PATH "${NURIKIT_PYTHON_MODULE}" NURIKIT_PYTHON_MODULE)
  string(REPLACE "/" "_" NURIKIT_PYTHON_MODULE "${NURIKIT_PYTHON_MODULE}")
  string(REGEX REPLACE
    "\.cpp$" ""
    NURIKIT_PYTHON_MODULE "${NURIKIT_PYTHON_MODULE}")
  string(PREPEND NURIKIT_PYTHON_MODULE "nurikit_python_")

  pybind11_add_module(
    "${NURIKIT_PYTHON_MODULE}"
    OPT_SIZE
    "${NURIKIT_PYTHON_SOURCE}")
  target_link_libraries("${NURIKIT_PYTHON_MODULE}"
    PRIVATE nuri_lib absl::strings)
  set_target_properties("${NURIKIT_PYTHON_MODULE}"
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/${NURIKIT_PYTHON_SOURCE_DIR_INV}${CMAKE_INSTALL_LIBDIR}"
    OUTPUT_NAME "${NURIKIT_PYTHON_MODULE_NAME}")

  install(TARGETS "${NURIKIT_PYTHON_MODULE}"
    LIBRARY
    DESTINATION "${NURIKIT_PYTHON_SOURCE_DIR}")

  add_dependencies(nuri_python "${NURIKIT_PYTHON_MODULE}")
endforeach()

add_subdirectory(docs EXCLUDE_FROM_ALL)
