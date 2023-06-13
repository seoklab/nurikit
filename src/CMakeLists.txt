#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

find_or_fetch_eigen()

include(FetchContent)

FetchContent_Declare(
  boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.xz
)
set(Boost_INCLUDE_LIBRARIES spirit)
set(Boost_ENABLE_CMAKE ON)
set(Boost_USE_STATIC_LIBS ON)
nurikit_make_available_deponly(boost)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

add_executable(nuri_exe "main.cpp")
set_target_properties(nuri_exe
  PROPERTIES
  OUTPUT_NAME nuri
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS nuri_exe RUNTIME)

file(GLOB_RECURSE NURI_SRCS *.cpp)
list(FILTER NURI_SRCS EXCLUDE REGEX "main\.cpp$")

if(NURI_SRCS)
  add_library(nuri_lib SHARED "${NURI_SRCS}")
  set_target_properties(nuri_lib PROPERTIES OUTPUT_NAME nuri)
  target_link_libraries(nuri_lib
    PUBLIC
    absl::strings absl::flat_hash_map absl::absl_check)

  target_system_include_directories(nuri_lib
    Eigen3::Eigen Boost::spirit Boost::fusion Boost::mpl Boost::optional)
  target_link_libraries(nuri_lib
    PRIVATE
    Boost::spirit Boost::fusion Boost::mpl Boost::optional)

  install(TARGETS nuri_lib LIBRARY)

  target_link_libraries(nuri_exe PRIVATE nuri_lib)
endif()
