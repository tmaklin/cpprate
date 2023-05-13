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
#include "lowrank_unit_test.hpp"

#include <cmath>

#include "CppRateRes.hpp"

TEST_F(LowrankTest, covariance_matrix) {
    const Eigen::MatrixXd &cov_got = covariance_matrix(this->f_draws);

    // Check dimensions of `beta_draws_got`
    EXPECT_EQ(cov_got.rows(), this->n_obs);
    EXPECT_EQ(cov_got.cols(), this->n_obs);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_obs; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    EXPECT_NEAR(cov_got(i, j), cov_f_draws_expected[i*this->n_obs + j], this->test_tolerance);
	}
    }
}

TEST_F(LowrankTest, project_f_draws) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);
    const Eigen::MatrixXd &project_f_draws_got = project_f_draws(this->f_draws, this->svd_design_mat_u);

    // Check dimensions of `project_f_draws_got`
    EXPECT_EQ(project_f_draws_got.rows(), this->num_nonzero_dims_expected);
    EXPECT_EQ(project_f_draws_got.cols(), this->num_nonzero_dims_expected);

    // Check contents of `beta_draws_got`
    for (size_t j = 0; j < this->num_nonzero_dims_expected; ++j) {
	for (size_t i = j; i < this->num_nonzero_dims_expected; ++i) {
	    EXPECT_NEAR(std::abs(project_f_draws_got(i, j)), std::abs(project_f_draws_expected[i*this->num_nonzero_dims_expected + j]), this->test_tolerance);
	}
    }
}

TEST_F(LowrankTest, approximate_cov_beta) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);
    const Eigen::SparseMatrix<double> &proj_f_draws = project_f_draws(this->f_draws, this->svd_design_mat_u);

    const Eigen::MatrixXd &cov_beta_got = approximate_cov_beta(proj_f_draws, this->svd_design_mat_v);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(cov_beta_got.rows(), this->n_design_dim);
    EXPECT_EQ(cov_beta_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(std::abs(cov_beta_got(i, j)), std::abs(approximate_cov_beta_expected[i*this->n_design_dim + j]), this->test_tolerance);
	}
    }
}

TEST_F(LowrankTest, approximate_beta_means) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);

    const Eigen::VectorXd &col_means_got = approximate_beta_means(this->f_draws, this->svd_design_mat_u, this->svd_design_mat_v);

    // Check dimensions of `col_means_got`
    EXPECT_EQ(col_means_got.size(), this->n_design_dim);

    // Check contents of `col_means_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(col_means_got(i), approximate_beta_means_expected[i], this->test_tolerance);
    }
}

TEST_F(LowrankTest, decompose_covariance_approximation) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);
    const Eigen::SparseMatrix<double> &proj_f_draws = project_f_draws(this->f_draws, this->svd_design_mat_u);

    const Eigen::MatrixXd svd_cov_u_got = decompose_covariance_approximation(proj_f_draws, this->svd_design_mat_v, this->rank_r);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(svd_cov_u_got.rows(), this->n_design_dim);
    EXPECT_EQ(svd_cov_u_got.cols(), this->num_nonzero_dims_expected);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->num_nonzero_dims_expected; ++j) {
	    EXPECT_NEAR(std::abs(svd_cov_u_got(i, j)), std::abs(proj_f_draws_svd_u_expected[i*this->num_nonzero_dims_expected + j]), this->test_tolerance);
	}
    }
}

TEST_F(LowrankTest, create_lambda) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);
    const Eigen::SparseMatrix<double> proj_f_draws = project_f_draws(this->f_draws, this->svd_design_mat_u);
    const Eigen::MatrixXd svd_cov_u = decompose_covariance_approximation(proj_f_draws, this->svd_design_mat_v, this->rank_r);

    const Eigen::MatrixXd lambda_got = create_lambda(svd_cov_u.adjoint());

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(lambda_got.rows(), this->n_design_dim);
    EXPECT_EQ(lambda_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(std::abs(lambda_got(i, j)), std::abs(lambda_expected[i*this->n_design_dim + j]), this->test_tolerance);
	}
    }
}
