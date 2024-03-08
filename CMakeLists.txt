cmake_minimum_required(VERSION 3.1)
project(cpprate)

set(CMAKE_CXX_STANDARD 17)

## Determine build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) ## Build Release by default
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -Wall -Wextra -Wpedantic")
endif()
if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -DNDEBUG -O3")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -DNDEBUG -O3")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-fast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")

if(CMAKE_BUILD_WITH_FLTO)
  cmake_policy(SET CMP0069 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION 1)
endif()

# Check dependencies
## OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CPPRATE_OPENMP_SUPPORT 1)
else()
  message(WARNING "OpenMP not found: parallelization will only be possible over SNPs (the `--threads-per-snp` argument will be ignored).")
  set(CPPRATE_OPENMP_SUPPORT 0)
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/cpprate_openmp_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/include/cpprate_openmp_config.hpp @ONLY)

## BLAS
find_package(BLAS)
if (BLAS_FOUND)
  set(CPPRATE_BLAS_SUPPORT 1)
else()
  set(CPPRATE_BLAS_SUPPORT 0)
  message(WARNING "BLAS not found: some matrix operations may run slower.")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/cpprate_blas_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/include/cpprate_blas_config.hpp @ONLY)

## eigen
if (DEFINED CMAKE_EIGEN_HEADERS)
  message(STATUS "eigen headers provided in: ${CMAKE_EIGEN_HEADERS}")
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/CMakeLists-eigen.txt.in ${CMAKE_CURRENT_BINARY_DIR}/external/eigen-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/eigen-download )
  if(result)
    message(FATAL_ERROR "CMake step for eigen failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/eigen-download )
  if(result)
    message(FATAL_ERROR "Build step for eigen failed: ${result}")
  endif()
  set(CMAKE_EIGEN_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/external/eigen)
endif()
include_directories(${CMAKE_EIGEN_HEADERS})

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
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/design_matrix_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/svd_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/fullrank_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/lowrank_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/kld_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/transformed_results_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/lowrank_integration_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/fullrank_integration_test.cpp)
  add_test(runTests runTests)
  target_link_libraries(runTests gtest gtest_main)
  if (OPENMP_FOUND)
      target_link_libraries(runTests OpenMP::OpenMP_CXX)
    endif()
  if (BLAS_FOUND)
    target_link_libraries(runTests ${BLAS_LIBRARIES})
  endif()
endif()

## bxzstr
if (DEFINED CMAKE_BXZSTR_HEADERS)
  message(STATUS "bxzstr headers provided in: ${CMAKE_BXZSTR_HEADERS}")
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/CMakeLists-bxzstr.txt.in ${CMAKE_CURRENT_BINARY_DIR}/external/bxzstr-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/bxzstr-download )
  if(result)
    message(FATAL_ERROR "CMake step for bxzstr failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/bxzstr-download )
  if(result)
    message(FATAL_ERROR "Build step for bxzstr failed: ${result}")
  endif()
  set(CMAKE_BXZSTR_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/external/bxzstr/include)
endif()
include_directories(${CMAKE_BXZSTR_HEADERS})

if(CMAKE_BUILD_EXECUTABLE)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
  add_executable(cpprate ${CMAKE_CURRENT_SOURCE_DIR}/src/cpprate.cpp)
  if (OPENMP_FOUND)
    target_link_libraries(cpprate OpenMP::OpenMP_CXX)
  endif()
  if (BLAS_FOUND)
    target_link_libraries(cpprate ${BLAS_LIBRARIES})
  endif()
  ## Check supported compression types
  find_package(BZip2)
  if (BZIP2_FOUND)
    include_directories(${BZIP2_INCLUDE_DIRS})
    target_link_libraries(cpprate ${BZIP2_LIBRARIES})
  endif()
  find_package(LibLZMA)
  if (LIBLZMA_FOUND)
    include_directories(${LIBLZMA_INCLUDE_DIRS})
    target_link_libraries(cpprate ${LIBLZMA_LIBRARIES})
  endif()
  find_package(ZLIB)
  if (ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    target_link_libraries(cpprate ${ZLIB_LIBRARIES})
  endif()
endif()

## cxxargs
if (DEFINED CMAKE_CXXARGS_HEADERS)
  message(STATUS "cxxargs headers provided in: ${CMAKE_CXXARGS_HEADERS}")
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/CMakeLists-cxxargs.txt.in ${CMAKE_BINARY_DIR}/external/cxxargs-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external/cxxargs-download )
  if(result)
    message(FATAL_ERROR "CMake step for cxxargs failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external/cxxargs-download )
  if(result)
    message(FATAL_ERROR "Build step for cxxargs failed: ${result}")
  endif()
  set(CMAKE_CXXARGS_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/external/cxxargs/include)
endif()
include_directories(${CMAKE_CXXARGS_HEADERS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/RedSVD)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/bs_thread_pool/include)
