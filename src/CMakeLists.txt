#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

add_compile_options(-pedantic -Wall -Wextra)

add_executable(NuriExe "main.cpp")
set_target_properties(NuriExe PROPERTIES OUTPUT_NAME nuri)
install(TARGETS NuriExe RUNTIME)

file(GLOB NURIKIT_SRCS *.cpp)
list(FILTER NURIKIT_SRCS EXCLUDE REGEX "main\.cpp$")

if(NURIKIT_SRCS)
  add_library(NuriLib SHARED "${NURIKIT_SRCS}")
  set_target_properties(NuriLib PROPERTIES OUTPUT_NAME nuri)
  install(TARGETS NuriLib LIBRARY)

  target_link_libraries(NuriExe PRIVATE NuriLib)
endif()
