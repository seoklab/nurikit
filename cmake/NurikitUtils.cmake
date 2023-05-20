macro(_nurikit_get_git_version_impl)
  find_package(Git)

  if(NOT Git_FOUND)
    message(WARNING "Git not found!")
    return()
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE nurikit_revision
    ERROR_QUIET)
  string(STRIP "${nurikit_revision}" nurikit_revision)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match --abbrev=0
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE NURIKIT_REF
    ERROR_QUIET)

  if(git_result EQUAL 0)
    string(STRIP "${NURIKIT_REF}" NURIKIT_REF)
    string(REPLACE v "" NURIKIT_VERSION "${NURIKIT_REF}")
    message(STATUS "Nurikit version from git: ${NURIKIT_VERSION}")
  else()
    execute_process(
      COMMAND ${GIT_EXECUTABLE} symbolic-ref -q --short HEAD
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE NURIKIT_REF
      ERROR_QUIET)

    if(NOT git_result EQUAL 0)
      set(NURIKIT_REF "${nurikit_revision}")
    endif()

    string(STRIP "${NURIKIT_REF}" NURIKIT_REF)
  endif()
endmacro()

function(nurikit_get_version)
  _nurikit_get_git_version_impl()

  if(NURIKIT_REF)
    message(STATUS "Nurikit ref from git: ${NURIKIT_REF}")
  else()
    message(NOTICE "Nurikit ref not found! Using unknown.")
    set(nurikit_revision "unknown")
    set(NURIKIT_REF "unknown")
  endif()

  if(NURIKIT_VERSION)
    set(NURIKIT_FULL_VERSION "${NURIKIT_VERSION}")
  else()
    message(NOTICE "Nurikit version not found! Using 0.1.0")
    set(NURIKIT_VERSION "0.1.0")
    set(NURIKIT_FULL_VERSION "${NURIKIT_VERSION}.dev0+${nurikit_revision}")
  endif()

  string(TIMESTAMP NURIKIT_YEAR "%Y" UTC)
  set(NURIKIT_YEAR "${NURIKIT_YEAR}" PARENT_SCOPE)
  set(NURIKIT_VERSION "${NURIKIT_VERSION}" PARENT_SCOPE)
  set(NURIKIT_FULL_VERSION "${NURIKIT_FULL_VERSION}" PARENT_SCOPE)
  set(NURIKIT_REF "${NURIKIT_REF}" PARENT_SCOPE)
endfunction()
