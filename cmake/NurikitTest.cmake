#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET nuri_all_test)
  add_custom_target(nuri_all_test)

  if(BUILD_TESTING)
    set_target_properties(nuri_all_test PROPERTIES EXCLUDE_FROM_ALL OFF)
  endif()
endif()
