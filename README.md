# cpprate - Variable Selection in Black Box Methods with RelATive cEntrality (RATE) Measures
cpprate is a reimplementation of https://github.com/lorinanthony/rate in C++.

# Installation
## Compiling from source
### Dependencies
- cmake >= v3.1
- C++11 compliant compiler
- git >= v2.0.0

# Building and running tests
Build tests with
```
mkdir build
cd build
cmake -DMCAKE_BUILD_TESTS=1 ..
make -j
```

Run tests with
```
bin/runTests
```

# BLAS
It's *highly* recommended to have BLAS installed on the system. Not having BLAS will seriously impact multithreading performance.

# License
cpprate is licensed under the [BSD-3-Clause license](https://opensource.org/licenses/BSD-3-Clause). A copy of the license is supplied with the project, or can alternatively be obtained from [https://opensource.org/licenses/BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause).
