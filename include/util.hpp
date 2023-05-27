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
#ifndef CPPRATE_UTIL_HPP
#define CPPRATE_UTIL_HPP

#include <cstddef>
#include <vector>

#include "cpprate_blas_config.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

inline Eigen::MatrixXd nonlinear_coefficients(const Eigen::SparseMatrix<double> &design_matrix, const Eigen::MatrixXd &f_draws) {
    const Eigen::MatrixXd &inv_X = Eigen::MatrixXd(design_matrix).completeOrthogonalDecomposition().pseudoInverse();
    return f_draws*inv_X.transpose();

}

inline Eigen::VectorXd col_means(const Eigen::MatrixXd &mat) {
    return mat.colwise().mean();
}


inline Eigen::MatrixXd covariance_matrix(const Eigen::MatrixXd &in) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(in.cols(), in.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate((in.rowwise() - in.colwise().mean()).transpose());
    tmp.array() /= double(in.rows() - 1);
    tmp.template triangularView<Eigen::Upper>() = tmp.transpose();
    return tmp;
}

inline Eigen::MatrixXd create_lambda(const Eigen::MatrixXd &U) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(U.cols(), U.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(U.transpose());
    tmp.template triangularView<Eigen::Upper>() = tmp.transpose();
    return tmp;
}

template <typename T>
inline Eigen::MatrixXd vec_to_dense_matrix(const std::vector<T> &vec, const size_t n_rows, const size_t n_cols) {
    // TODO tests
    Eigen::MatrixXd mat(n_rows, n_cols);
    for (size_t i = 0; i < n_rows; ++i) {
	for (size_t j = 0; j < n_cols; ++j) {
	    mat(i, j) = (T)vec[i*n_cols + j];
	}
    }
    return mat;
}

template <typename T, typename V>
Eigen::SparseMatrix<T> vec_to_sparse_matrix(const std::vector<V> &vec, const size_t n_rows, const size_t n_cols) {
    std::vector<Eigen::Triplet<T>> triplet_list;
    triplet_list.reserve(n_rows*n_cols);

    for (size_t i = 0; i < n_rows; ++i) {
	for (size_t j = 0; j < n_cols; ++j) {
	    if (vec[i*n_cols + j] != (V)0) {
		T val = (T)vec[i*n_cols + j];
		triplet_list.emplace_back(Eigen::Triplet<T>(i, j, val));
	    }
	}
    }

    Eigen::SparseMatrix<T> mat(n_rows, n_cols);
    mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return mat;
}

#endif
