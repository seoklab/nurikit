#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/nuri"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT Development
  FILES_MATCHING PATTERN "*.h")
