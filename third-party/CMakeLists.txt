set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory("abseil-cpp")

if(NURIKIT_BUILD_TESTS)
  add_subdirectory("googletest")
endif()
