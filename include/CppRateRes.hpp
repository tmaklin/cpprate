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

#include <Eigen/SparseCore>

#include "RedSVD.h"

#include <vector>
#include <cstddef>
#include <cmath>
#include <iostream>

struct CppRateRes {
public:
    double ESS = 0.0;
    double Delta = 0.0;
    std::vector<double> RATE = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> KLD = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
};

template <typename T, typename V>
Eigen::SparseMatrix<T> sparsify_design_matrix(const size_t n_rows, const size_t n_cols, const std::vector<V> &vals) {
    std::vector<Eigen::Triplet<T>> triplet_list;
    triplet_list.reserve(n_rows*n_cols);

    for (size_t i = 0; i < n_rows; ++i) {
	for (size_t j = 0; j < n_cols; ++j) {
	    if (vals[i*n_cols + j] != (V)0) {
		T val = (T)vals[i*n_cols + j];
		triplet_list.emplace_back(Eigen::Triplet<T>(i, j, val));
	    }
	}
    }
    Eigen::SparseMatrix<T> mat(n_rows, n_cols);
    mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return mat;
}

inline void RATE(const size_t n_obs, const size_t n_snps, const std::vector<bool> &design_matrix) {
    // ## WARNING: Do not compile with -ffast-math

    const size_t svd_rank = std::min(n_obs, n_snps);
    const double prop_var = 1.0;

    const Eigen::SparseMatrix<double> &X = sparsify_design_matrix<double, bool>(n_obs, n_snps, design_matrix);

    RedSVD::RedSVD<Eigen::SparseMatrix<double>> res(X, svd_rank);

    std::vector<double> px(svd_rank);
    std::vector<bool> r_X(svd_rank);

    double sv_sum = 0.0;
    for (size_t i = 0; i < svd_rank; ++i) {
	sv_sum += res.singularValues()[i]*res.singularValues()[i];
    }

    size_t num_r_X_set = 0;

    for (size_t i = 0; i < svd_rank; ++i) {
	px[i] = res.singularValues()[i]*res.singularValues()[i];
	if (i > 0) {
	    px[i] += px[i - 1];
	}
	px[i] /= sv_sum;
	r_X[i] = (res.singularValues()[i] > (double)1e-10) && (px[i] < prop_var);
	num_r_X_set += r_X[i];
    }

}

#endif
