if(NOT TARGET nurikit_all_test)
  add_custom_target(nurikit_all_test)

  if(BUILD_TESTING)
    set_target_properties(
      nurikit_all_test PROPERTIES EXCLUDE_FROM_ALL OFF)
  endif()
endif()

include(GoogleTest)

function(nurikit_add_single_test name)
  set(target_name nurikit_${name})
  add_executable("${target_name}" ${name}.cpp)
  target_link_libraries("${target_name}"
    PRIVATE GTest::gtest GTest::gtest_main absl::absl_log absl::absl_check)

  if(TARGET nuri_lib)
    target_link_libraries("${target_name}" PRIVATE nuri_lib)
  endif()

  gtest_discover_tests("${target_name}" "" AUTO)
  add_dependencies(nurikit_all_test "${target_name}")
endfunction()

function(nurikit_add_test)
  foreach(name ${ARGN})
    nurikit_add_single_test(${name})
  endforeach()
endfunction()
