#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

add_compile_options(-pedantic -Wall -Wextra)

add_executable(nurikit_exe "main.cpp")
set_target_properties(nurikit_exe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

install(TARGETS nurikit_exe RUNTIME)

file(GLOB NURIKIT_SRCS *.cpp)
list(FILTER NURIKIT_SRCS EXCLUDE REGEX "main\.cpp$")

if(NURIKIT_SRCS)
  add_library(nurikit_lib SHARED "${NURIKIT_SRCS}")
  target_link_libraries(nurikit_exe PRIVATE nurikit_lib)

  install(TARGETS nurikit_lib LIBRARY)
endif()
