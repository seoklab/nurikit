set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory("abseil-cpp")

if(BUILD_TESTING)
  add_subdirectory("googletest")
endif()
