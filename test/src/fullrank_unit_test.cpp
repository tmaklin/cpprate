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
#include "fullrank_unit_test.hpp"

#include <cmath>

#include <Eigen/Dense>

#include "CppRateRes.hpp"

TEST_F(FullrankTest, nonlinear_coefficients) {
    const Eigen::MatrixXd &beta_draws_got = nonlinear_coefficients(this->sparse_design_matrix, this->f_draws);

    // Check dimensions of `beta_draws_got`
    EXPECT_EQ(beta_draws_got.rows(), this->n_f_draws);
    EXPECT_EQ(beta_draws_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(beta_draws_got(i, j), beta_draws_expected[i*this->n_design_dim + j], this->test_tolerance);
	}
    }
}

TEST_F(FullrankTest, col_means) {
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(this->sparse_design_matrix, this->f_draws);

    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);

    // Check dimensions
    EXPECT_EQ(col_means_beta.size(), this->n_design_dim);

    // Check contesnts of `col_means_beta`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(col_means_beta[i], this->col_means_beta_expected[i], 1e-6);
    }
}

TEST_F(FullrankTest, covariance_matrix) {
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(this->sparse_design_matrix, this->f_draws);
    const Eigen::MatrixXd &cov_got = covariance_matrix(beta_draws);

    // Check dimensions of `beta_draws_got`
    EXPECT_EQ(cov_got.rows(), this->n_design_dim);
    EXPECT_EQ(cov_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(cov_got(i, j), beta_draws_cov_expected[i*this->n_design_dim + j], this->test_tolerance);
	}
    }
}

TEST_F(FullrankTest, decompose_covariance_matrix) {
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(this->sparse_design_matrix, this->f_draws);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);

    const Eigen::MatrixXd &cov_svd_u_got = decompose_covariance_matrix(cov);

    // Check dimensions of `cov_svd_u_got`
    EXPECT_EQ(cov_svd_u_got.rows(), this->cov_svd_u_nonzero_dims);
    EXPECT_EQ(cov_svd_u_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->cov_svd_u_nonzero_dims; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(cov_svd_u_got(i, j)), std::abs(cov_svd_u_expected[i*this->n_design_dim + j]), this->test_tolerance);
	}
    }
}

TEST_F(FullrankTest, create_lambda) {
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(this->sparse_design_matrix, this->f_draws);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);
    const Eigen::MatrixXd &svd_cov_u = decompose_covariance_matrix(cov);

    const Eigen::MatrixXd &lambda_got = create_lambda(svd_cov_u);

    // Check dimensions of `lambda_got`
    EXPECT_EQ(lambda_got.rows(), this->n_design_dim);
    EXPECT_EQ(lambda_got.cols(), this->n_design_dim);

    // Check contents of `lambda_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(lambda_got(i, j)), std::abs(lambda_expected[i*this->n_design_dim + j]), 1e-6);
	}
    }
}
