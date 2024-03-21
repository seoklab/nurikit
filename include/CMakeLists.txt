#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

configure_file(nuri/nurikit_config.h.in
  generated/include/nuri/nurikit_config.h @ONLY)
list(APPEND NURI_INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
set(NURI_INCLUDE_DIRECTORIES ${NURI_INCLUDE_DIRECTORIES} PARENT_SCOPE)

if(NOT SKBUILD)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include/nuri/nurikit_config.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nuri")

  install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/nuri"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h")
endif()
