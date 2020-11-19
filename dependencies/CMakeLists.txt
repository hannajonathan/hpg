include(FetchContent)

FetchContent_declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG master)

add_subdirectory(googletest)
