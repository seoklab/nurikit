file(GLOB PRJ_SRCS *.cpp)
list(FILTER PRJ_SRCS EXCLUDE REGEX "main\.cpp$")
add_library(project_lib STATIC "${PRJ_SRCS}")

add_executable(project_exe "main.cpp")
target_link_libraries(project_exe PRIVATE project_lib)
set_target_properties(project_exe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

install(TARGETS project_exe RUNTIME)
