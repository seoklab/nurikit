#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

# Don't fail build on warnings from third-party code.
add_compile_options(-Wno-error)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_USE_SYSTEM_INCLUDES ON)
add_subdirectory("abseil-cpp")

add_subdirectory("googletest")
