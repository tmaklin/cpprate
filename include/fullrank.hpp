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
#ifndef CPPRATE_FULLRANK_HPP
#define CPPRATE_FULLRANK_HPP

#include <cstddef>
#include <cmath>
#include <vector>

#include "cpprate_openmp_config.hpp"
#include "cpprate_blas_config.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "svd_wrapper.hpp"
#include "RATE_res.hpp"
#include "util.hpp"

inline Eigen::MatrixXd sherman_r(const Eigen::MatrixXd &ap, const Eigen::VectorXd &u) {
    const Eigen::MatrixXd &tmp = u * u.transpose() * ap;
    return (ap - ( (ap * tmp).array() / (1 + (tmp).array())).matrix());
}

inline double dropped_predictor_kld(const Eigen::MatrixXd &lambda, const Eigen::VectorXd &cov_beta_col, const double mean_beta, const size_t predictor_id) {
    double log_m = std::log(std::abs(mean_beta));
    const Eigen::MatrixXd &U_Lambda_sub = sherman_r(lambda, cov_beta_col);

    double alpha = 0.0;

    for (size_t k = 0; k < U_Lambda_sub.cols(); k++) {
	if (k != predictor_id) {
	    double max_element = 0.0;
	    for (size_t j = 0; j < U_Lambda_sub.rows(); ++j) {
		if (j != predictor_id) {
		    max_element = (max_element >= std::log(1e-16 + std::abs(U_Lambda_sub(j, predictor_id))) + std::log(1e-16 + std::abs(U_Lambda_sub(j, k))) ? max_element : std::log(1e-16 + std::abs(U_Lambda_sub(j, predictor_id))) + std::log(1e-16 + std::abs(U_Lambda_sub(j, k))));
		}
	    }

	    double tmp_sum = 0.0;
	    for (size_t j = 0; j < U_Lambda_sub.rows(); ++j) {
		if (j != predictor_id) {
		    tmp_sum += std::exp(std::log(1e-16 + std::abs(U_Lambda_sub(j, predictor_id))) + std::log(1e-16 + std::abs(U_Lambda_sub(j, k))) - max_element);
		}
	    }
	    tmp_sum = std::log(tmp_sum) + max_element;
	    alpha += std::exp(tmp_sum + std::log(1e-16 + std::abs(U_Lambda_sub(k, predictor_id))));
	}
    }

    return std::log(0.5) + log_m + std::log(alpha) + log_m;
}

inline Eigen::MatrixXd decompose_covariance_matrix(const Eigen::MatrixXd &covariance_matrix) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    const Eigen::MatrixXd &inv_cov_mat = covariance_matrix.completeOrthogonalDecomposition().pseudoInverse();
    const size_t rank = inv_cov_mat.cols();

    Eigen::VectorXd svd_singular_values;
    Eigen::MatrixXd svd_U;

    {
	Eigen::MatrixXd svd_V;

	svd<Eigen::MatrixXd>(inv_cov_mat, rank, &svd_U, &svd_V, &svd_singular_values);
    }

    std::vector<bool> r_D(rank);
    size_t num_r_D_set = 0;
#pragma omp parallel for schedule(static) reduction(+:num_r_D_set)
    for (size_t i = 0; i < rank; ++i) {
	r_D[i] = svd_singular_values[i] > 1e-10;
	num_r_D_set += r_D[i];
    }

    size_t n_rows_D = svd_U.rows();
    size_t n_cols_D = svd_U.cols();
    Eigen::MatrixXd u(num_r_D_set, n_rows_D);
    for (size_t i = 0; i < num_r_D_set; ++i) {
	for (size_t j = 0; j < n_rows_D; ++j) {
	    u(i, j) = std::sqrt(svd_singular_values[i])*svd_U(j, i);
	}
    }

    return u;
}

inline RATEd RATE_fullrank(const Eigen::MatrixXd &f_draws, const Eigen::SparseMatrix<double> &design_matrix, const size_t n_snps) {
    // ## WARNING: Do not compile with -ffast-math

    Eigen::MatrixXd beta_draws = std::move(nonlinear_coefficients(design_matrix, f_draws));

    const Eigen::MatrixXd &cov_beta = covariance_matrix(beta_draws);
    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);
    const Eigen::MatrixXd &Lambda = create_lambda(decompose_covariance_matrix(cov_beta));

    std::vector<double> log_KLD(n_snps);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_snps; ++i) {
	log_KLD[i] = dropped_predictor_kld(Lambda, cov_beta.col(i), col_means_beta[i], i);
    }

    return RATEd(log_KLD);
}

#endif
