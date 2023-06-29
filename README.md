# cpprate - Variable Selection in Black Box Methods with RelATive cEntrality (RATE) Measures
cpprate is a reimplementation of https://github.com/lorinanthony/rate in C++.

# Installation
This is a header-only library so simply include "CppRateRes.hpp" or
"CppRateRes_mpi.hpp" in your project. An optional command-line
executable is also provided, see instructions below for compiling.
## Compiling the cpprate executable from source
### Dependencies
- cmake >= v3.1
- C++11 compliant compiler
- git >= v2.0.0

Build the `cpprate` executable with
```
mkdir build
cd build
cmake -DMCAKE_BUILD_EXECUTABLE=1 ..
make -j
```

# Usage
## Nonlinear coefficients
Supply the design matrix as a `n_observations`x`n_snps` comma-separated matrix via the `-x` argument and the nonlinear `n_posterior_draws`x`n_observations` comma-separated matrix via the `-f` argument by running
```
cpprate -x design_matrix_file.csv -f nonlinear_draws_file.csv
```
this will run the lowrank approximation by default. To run the fullrank approximation, use the `--fullrank` toggle (note: can be slow).

## Linear coefficients
Supply the posterior draws for the linear coefficients as a `n_posterior_draws`x`n_snps` comma-separated matrix via the `--beta-draws` argument by running
```
cpprate --beta-draws linear_coefficients_file.csv
```
this will run the fullrank algorithm because the lowrank approximation is only valid for nonlinear coefficients.

## Parallelization
Add the number of threads via the `-t` argument to parallelize calculation over the number of snps. Beware: adding threads will increase the memory consumption roughly linearly but results in a roughly linear speedup.

For more fine-grained parallelization over both the number of snps and within each snp, use the MPI implementation, where the `-t` argument adds threads for each MPI process parallelizing calculation over the snps. Note that adding threads within each snp is only useful for very large inputs.

# Development
## Building tests
Build tests with
```
mkdir build
cd build
cmake -DMCAKE_BUILD_TESTS=1 ..
make -j
```

## Running tests
Run tests with
```
bin/runTests
```

# License
cpprate is licensed under the [BSD-3-Clause license](https://opensource.org/licenses/BSD-3-Clause). A copy of the license is supplied with the project, or can alternatively be obtained from [https://opensource.org/licenses/BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause).
