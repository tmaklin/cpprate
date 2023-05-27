// cpprate: Variable Selection in Black Box Methods with RelATive cEntrality (RATE) Measures
// https://github.com/tmaklin/cpprate
// Copyright (c) 2023 Tommi Mäklin (tommi@maklin.fi)
//
// BSD-3-Clause license
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     (1) Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//     (2) Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//
//     (3)The name of the author may not be used to
//     endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
#ifndef CPPRATE_CPPRATERES_HPP
#define CPPRATE_CPPRATERES_HPP

#include <vector>
#include <cstddef>

#include "cpprate_openmp_config.hpp"
#include "cpprate_blas_config.hpp"

#include <Eigen/Dense>

inline std::vector<double> get_col(const std::vector<double> &flat, const size_t n_rows, const size_t n_cols, const size_t col_id) {
    std::vector<double> res(n_rows);

#pragma omp parallel for schedule(static)
    for (size_t i = col_id; i < n_cols; ++i) {
	size_t pos_in_lower_tri = col_id * n_rows + i - col_id * (col_id - 1)/2 - col_id;
	res[i] = flat[pos_in_lower_tri];
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < col_id; ++i) {
	size_t pos_in_lower_tri = i * n_rows + col_id - i * (i - 1)/2 - i;
	res[i] = flat[pos_in_lower_tri];
    }

    return res;
}

inline std::vector<double> flatten_lambda(const Eigen::MatrixXd &Lambda) {
    // TODO tests
    size_t dim = Lambda.rows();
    std::vector<double> flat_lambda(dim * (dim + 1)/2, 0.0);

#pragma omp parallel for schedule(guided)
    for (int64_t j = dim - 1; j >= 0; --j) {
	size_t col_start = j * dim - j * (j - 1)/2 - j;
	for (size_t i = j; i < dim; ++i) {
	    flat_lambda[col_start + i] = Lambda(i, j);
	}
    }

    return flat_lambda;
}

#endif
