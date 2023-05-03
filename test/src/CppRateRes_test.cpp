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
#include "CppRateRes_test.hpp"

#include <cmath>

#include "CppRateRes.hpp"

// Test cpprate
TEST_F(DesignMatrixTest, sparsify_design_matrix) {
    const Eigen::SparseMatrix<double> &mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    for (size_t i = 0; i < this->n_obs; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_EQ(this->design_matrix[i*n_design_dim + j], mat.coeff(i, j));
	}
    }
}

TEST_F(SvdTest, svd_factor_u) {
    Eigen::MatrixXd svd_u_got;
    Eigen::MatrixXd svd_v;
    Eigen::VectorXd svd_d;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    svd(this->sparse_design_matrix, svd_rank, &svd_u_got, &svd_v, &svd_d);

    // Check dimensions of `u`
    EXPECT_EQ(svd_rank, svd_u_got.rows());
    EXPECT_EQ(svd_rank, svd_u_got.cols());

    // Check contents of `u`
    for (size_t i = 0; i < svd_rank; ++i) {
	for (size_t j = 0; j < svd_rank; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_u_got(i, j)), std::abs(svd_u_expected[i*svd_rank + j]), this->test_tolerance);
	}
    }
}

TEST_F(SvdTest, svd_factor_v) {
    Eigen::MatrixXd svd_u;
    Eigen::MatrixXd svd_v_got;
    Eigen::VectorXd svd_d;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    svd(this->sparse_design_matrix, svd_rank, &svd_u, &svd_v_got, &svd_d);

    // Check dimensions of `v`
    EXPECT_EQ(this->n_design_dim, svd_v_got.rows());
    EXPECT_EQ(svd_rank, svd_v_got.cols());

    // Check contents of `v`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < svd_rank; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_v_got(i, j)), std::abs(svd_v_expected[i*svd_rank + j]), this->test_tolerance);
	}
    }
}

TEST_F(SvdTest, svd_singular_values) {
    Eigen::MatrixXd svd_u;
    Eigen::MatrixXd svd_v;
    Eigen::VectorXd svd_d_got;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    svd(this->sparse_design_matrix, svd_rank, &svd_u, &svd_v, &svd_d_got);

    // Check dimensions of `d`
    EXPECT_EQ(svd_rank, svd_d_got.size());

    // Check contents of `d`
    for (size_t i = 0; i < svd_rank; ++i) {
	EXPECT_NEAR(svd_d_got(i), svd_d_expected[i], this->test_tolerance);
    }
}

TEST_F(DecomposeDesignMatrixTest, svd_scaled_u) {
    Eigen::MatrixXd svd_u_got;
    Eigen::MatrixXd svd_v;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    decompose_design_matrix(this->sparse_design_matrix, svd_rank, this->prop_var, &svd_u_got, &svd_v);

    size_t num_nonzero_dims = svd_u_got.rows();

    // Check that the correct number of dimensions are dropped
    EXPECT_EQ(this->num_nonzero_dims_expected, num_nonzero_dims);

    // Check dimensions of `u`
    EXPECT_EQ(this->num_nonzero_dims_expected, svd_u_got.rows());
    EXPECT_EQ(this->n_obs, svd_u_got.cols());

    // Check contents of `u`
    for (size_t i = 0; i < this->num_nonzero_dims_expected; ++i) {
	for(size_t j = 0; j < this->n_obs; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_u_got(i, j)), std::abs(svd_u_expected[i*this->n_obs + j]), this->test_tolerance);
	}
    }
}

TEST_F(DecomposeDesignMatrixTest, svd_dropped_v) {
    Eigen::MatrixXd svd_u;
    Eigen::MatrixXd svd_v_got;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    decompose_design_matrix(this->sparse_design_matrix, svd_rank, this->prop_var, &svd_u, &svd_v_got);

    size_t num_nonzero_dims = svd_v_got.cols();

    // Check that the correct number of dimensions are dropped
    EXPECT_EQ(this->num_nonzero_dims_expected, num_nonzero_dims);

    // Check dimensions of `v`
    EXPECT_EQ(this->n_design_dim, svd_v_got.rows());
    EXPECT_EQ(this->num_nonzero_dims_expected, svd_v_got.cols());

    // Check contents of `v`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->num_nonzero_dims_expected; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_v_got(i, j)), std::abs(svd_v_expected[i*this->num_nonzero_dims_expected + j]), this->test_tolerance);
	}
    }
}

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
    const Eigen::MatrixXd project_f_draws_got = project_f_draws(this->f_draws, this->svd_design_mat_u);

    // Check dimensions of `project_f_draws_got`
    EXPECT_EQ(project_f_draws_got.rows(), this->num_nonzero_dims_expected);
    EXPECT_EQ(project_f_draws_got.cols(), this->num_nonzero_dims_expected);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->num_nonzero_dims_expected; ++i) {
	for (size_t j = 0; j < this->num_nonzero_dims_expected; ++j) {
	    EXPECT_NEAR(std::abs(project_f_draws_got(i, j)), std::abs(project_f_draws_expected[i*this->num_nonzero_dims_expected + j]), this->test_tolerance);
	}
    }
}

TEST_F(LowrankTest, approximate_cov_beta) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);
    const Eigen::MatrixXd &proj_f_draws = project_f_draws(this->f_draws, this->svd_design_mat_u);

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
    const Eigen::MatrixXd &proj_f_draws = project_f_draws(this->f_draws, this->svd_design_mat_u);

    const Eigen::MatrixXd svd_cov_u_got = decompose_covariance_approximation(proj_f_draws, this->svd_design_mat_v, this->rank_r);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(svd_cov_u_got.rows(), this->n_design_dim);
    EXPECT_EQ(svd_cov_u_got.cols(), this->num_nonzero_dims_expected);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->num_nonzero_dims_expected; ++j) {
	    EXPECT_NEAR(std::abs(svd_cov_u_got(i, j)), std::abs(proj_f_draws_svd_u_expected[i*this->rank_r + j]), this->test_tolerance);
	}
    }
}

TEST_F(LowrankTest, create_lambda) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(this->f_draws);
    const Eigen::MatrixXd proj_f_draws = project_f_draws(this->f_draws, this->svd_design_mat_u);
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

TEST_F(KldTest, sherman_r) {
    const Eigen::MatrixXd &sherman_r_got = sherman_r(lambda_fullrank, cov_fullrank.col(1), cov_fullrank.col(1));

    // Check dimensions of `sherman_r_got`
    EXPECT_EQ(sherman_r_got.rows(), this->n_design_dim);
    EXPECT_EQ(sherman_r_got.cols(), this->n_design_dim);

    // Check contents of `sherman_r_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(sherman_r_got(i, j), this->sherman_r_expected[i*this->n_design_dim + j], this->test_tolerance);
	}
    }
}

TEST_F(KldTest, dropped_predictor_kld) {
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(dropped_predictor_kld(lambda_fullrank, cov_fullrank, col_means_fullrank, i), this->expected_KLD[i], 1e-4);
    }
}

TEST_F(TransformedResultsTest, RATE) {
    const std::vector<double> &RATE_got = rate_from_kld(this->KLD, this->KLD_sum);

    // Check dimensions of `RATE_got`
    EXPECT_EQ(RATE_got.size(), this->n_design_dim);

    // Check contents of `RATE_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(RATE_got[i], this->RATE_expected[i], this->test_tolerance);
    }
}

TEST_F(TransformedResultsTest, Delta) {
    const std::vector<double> &RATE = rate_from_kld(this->KLD, this->KLD_sum);

    const double delta_got = rate_delta(RATE);
    EXPECT_NEAR(delta_got, this->Delta_expected, this->test_tolerance);
}

TEST_F(TransformedResultsTest, ESS) {
    const std::vector<double> &RATE = rate_from_kld(this->KLD, this->KLD_sum);
    const double delta = rate_delta(RATE);

    const double ess_got = delta_to_ess(delta);
    EXPECT_NEAR(ess_got, this->ESS_expected, this->test_tolerance);
}
