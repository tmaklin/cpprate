cmake_minimum_required(VERSION 3.1)
project(cpprate)

set (CMAKE_CXX_STANDARD 11)

## Download googletest if building tests
if(CMAKE_BUILD_TESTS)
  if (DEFINED CMAKE_GOOGLETEST_HEADERS)
    message(STATUS "googletest headers provided in: ${CMAKE_GOOGLETEST_HEADERS}")
  else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/CMakeLists-googletest.txt.in ${CMAKE_CURRENT_BINARY_DIR}/external/googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/googletest-download )
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/googletest-download )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()
    set(CMAKE_GOOGLETEST_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/external/googletest/googletest/include)
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/external/googletest
                     ${CMAKE_CURRENT_BINARY_DIR}/external/googletest/build)
  endif()
  set_target_properties(gmock PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(gmock_main PROPERTIES EXCLUDE_FROM_ALL 1)
  include_directories("${CMAKE_GOOGLETEST_HEADERS}")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/include)
  enable_testing()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
  add_executable(runTests
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/cpprate_integration_test.cpp)
  add_test(runTests runTests)
  target_link_libraries(runTests gtest gtest_main)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
