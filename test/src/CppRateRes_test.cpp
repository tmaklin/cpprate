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
TEST_F(CppRateResTest, sparsify_design_matrix) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    for (size_t i = 0; i < this->n_obs; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_EQ(this->design_matrix[i*n_design_dim + j], mat.coeff(i, j));
	}
    }
}

TEST_F(CppRateResTest, svd) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd u_got;
    Eigen::MatrixXd v_got;
    Eigen::VectorXd d_got;

    size_t svd_rank = this->n_obs;
    svd(mat, svd_rank, &u_got, &v_got, &d_got);

    // Check dimensions of `u`
    EXPECT_EQ(svd_rank, u_got.rows());
    EXPECT_EQ(svd_rank, u_got.cols());

    // Check dimensions of `v`
    EXPECT_EQ(this->n_design_dim, v_got.rows());
    EXPECT_EQ(svd_rank, v_got.cols());

    // Check dimensions of `d`
    EXPECT_EQ(svd_rank, d_got.size());

    // Check contents of `u`
    for (size_t i = 0; i < svd_rank; ++i) {
	for (size_t j = 0; j < svd_rank; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(u_got(i, j)), std::abs(svd_u[i*svd_rank + j]), 1e-6);
	}
    }

    // Check contents of `v`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < svd_rank; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(v_got(i, j)), std::abs(svd_v[i*svd_rank + j]), 1e-6);
	}
    }

    // Check contents of `d`
    for (size_t i = 0; i < svd_rank; ++i) {
	EXPECT_NEAR(d_got(i), svd_d[i], 1e-6);
    }
}

TEST_F(CppRateResTest, decompose_design_matrix) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd u_got;
    Eigen::MatrixXd v_got;
    size_t svd_rank = this->n_obs;
    decompose_design_matrix(mat, svd_rank, 1.0, &u_got, &v_got);

    size_t num_invariant_dims = u_got.rows();

    // Check that the correct number of dimensions are dropped
    EXPECT_EQ(this->num_expected_invariant_dims, num_invariant_dims);

    // Check dimensions of `u`
    EXPECT_EQ(this->num_expected_invariant_dims, u_got.rows());
    EXPECT_EQ(this->n_obs, u_got.cols());

    // Check dimensions of `v`
    EXPECT_EQ(this->n_design_dim, v_got.rows());
    EXPECT_EQ(this->num_expected_invariant_dims, v_got.cols());

    // Check contents of `u`
    for (size_t i = 0; i < num_expected_invariant_dims; ++i) {
	for(size_t j = 0; j < this->n_obs; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(u_got(i, j)), std::abs(svd_u_dropped[i*this->n_obs + j]), 1e-6);
	}
    }

    // Check contents of `v`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->num_expected_invariant_dims; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(v_got(i, j)), std::abs(svd_v_dropped[i*this->num_expected_invariant_dims + j]), 1e-6);
	}
    }
}

TEST_F(CppRateResTest, nonlinear_coefficients) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }

    const Eigen::MatrixXd &beta_draws_got = nonlinear_coefficients(mat, f_draws_mat);

    // Check dimensions of `beta_draws_got`
    EXPECT_EQ(beta_draws_got.rows(), this->n_f_draws);
    EXPECT_EQ(beta_draws_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(beta_draws_got(i, j), beta_draws_expected[i*this->n_design_dim + j], 1e-6);
	}
    }
}

TEST_F(CppRateResTest, covariance_matrix_beta_draws) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }

    const Eigen::MatrixXd &beta_draws_got = nonlinear_coefficients(mat, f_draws_mat);

    const Eigen::MatrixXd &cov_got = covariance_matrix(beta_draws_got);

    // Check dimensions of `beta_draws_got`
    EXPECT_EQ(cov_got.rows(), this->n_design_dim);
    EXPECT_EQ(cov_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(cov_got(i, j), beta_draws_cov_expected[i*this->n_design_dim + j], 1e-7);
	}
    }
}

TEST_F(CppRateResTest, covariance_matrix_f_draws) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }

    const Eigen::MatrixXd &cov_got = covariance_matrix(f_draws_mat);

    // Check dimensions of `beta_draws_got`
    EXPECT_EQ(cov_got.rows(), this->n_obs);
    EXPECT_EQ(cov_got.cols(), this->n_obs);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_obs; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    EXPECT_NEAR(cov_got(i, j), f_draws_cov_expected[i*this->n_obs + j], 1e-6);
	}
    }
}

TEST_F(CppRateResTest, project_cov_f_draws) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(f_draws_mat);

    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    size_t svd_rank = this->n_obs;
    decompose_design_matrix(mat, svd_rank, 1.0, &u, &v);

    const Eigen::MatrixXd project_cov_f_draws_got = project_cov_f_draws(f_draws_mat, u);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(project_cov_f_draws_got.rows(), this->num_expected_invariant_dims);
    EXPECT_EQ(project_cov_f_draws_got.cols(), this->num_expected_invariant_dims);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->num_expected_invariant_dims; ++i) {
	for (size_t j = 0; j < this->num_expected_invariant_dims; ++j) {
	    EXPECT_NEAR(std::abs(project_cov_f_draws_got(i, j)), std::abs(project_cov_f_draws_expected[i*this->num_expected_invariant_dims + j]), 1e-7);
	}
    }
}


TEST_F(CppRateResTest, approximate_cov_beta) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(f_draws_mat);

    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    size_t svd_rank = this->n_obs;
    decompose_design_matrix(mat, svd_rank, 1.0, &u, &v);
    const Eigen::MatrixXd &Sigma_star = project_cov_f_draws(f_draws_mat, u); // This works

    const Eigen::MatrixXd &cov_beta_got = approximate_cov_beta(Sigma_star, v);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(cov_beta_got.rows(), this->n_design_dim);
    EXPECT_EQ(cov_beta_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(std::abs(cov_beta_got(i, j)), std::abs(approximate_cov_beta_expected[i*this->n_design_dim + j]), 1e-7);
	}
    }
}

TEST_F(CppRateResTest, approximate_beta_means) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(f_draws_mat);

    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    size_t svd_rank = this->n_obs;
    decompose_design_matrix(mat, svd_rank, 1.0, &u, &v);

    const Eigen::VectorXd col_means_got = approximate_beta_means(f_draws_mat, u, v);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(col_means_got.size(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(col_means_got(i), approximate_beta_means_expected[i], 1e-6);
    }
}

TEST_F(CppRateResTest, decompose_covariance_approximation) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(f_draws_mat);

    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    size_t svd_rank = this->n_obs;
    decompose_design_matrix(mat, svd_rank, 1.0, &u, &v);
    const Eigen::MatrixXd project_cov_f_draws_mat = project_cov_f_draws(f_draws_mat, u);

    const Eigen::MatrixXd svd_cov_u_got = decompose_covariance_approximation(project_cov_f_draws_mat, v, 5);

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(svd_cov_u_got.rows(), this->n_design_dim);
    EXPECT_EQ(svd_cov_u_got.cols(), 5);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < 5; ++j) {
	    EXPECT_NEAR(std::abs(svd_cov_u_got(i, j)), std::abs(svd_project_cov_f_draws_u[i*5 + j]), 1e-7);
	}
    }
}

TEST_F(CppRateResTest, decompose_covariance_matrix) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);

    const Eigen::MatrixXd &svd_cov_u_got = decompose_covariance_matrix(cov);

    // Check dimensions of `svd_cov_u_got`
    EXPECT_EQ(svd_cov_u_got.rows(), this->svd_cov_u_invariant_dims);
    EXPECT_EQ(svd_cov_u_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->svd_cov_u_invariant_dims; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_cov_u_got(i, j)), std::abs(svd_cov_u_expected[i*this->n_design_dim + j]), 1e-6);
	}
    }
}

TEST_F(CppRateResTest, col_means) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);

    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);

    // Check dimensions
    EXPECT_EQ(col_means_beta.size(), this->n_design_dim);

    // Check contesnts of `col_means_beta`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(col_means_beta[i], this->col_means_beta_expected[i], 1e-6);
    }
}

TEST_F(CppRateResTest, create_lambda_fullrank) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
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

TEST_F(CppRateResTest, create_lambda_lowrank) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(f_draws_mat);

    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    size_t svd_rank = this->n_obs;
    decompose_design_matrix(mat, svd_rank, 1.0, &u, &v);
    const Eigen::MatrixXd project_cov_f_draws_mat = project_cov_f_draws(f_draws_mat, u);
    const Eigen::MatrixXd svd_cov_u = decompose_covariance_approximation(project_cov_f_draws_mat, v, 5);

    const Eigen::MatrixXd lambda_got = create_lambda(svd_cov_u.adjoint());

    // Check dimensions of `project_cov_f_draws_got`
    EXPECT_EQ(lambda_got.rows(), this->n_design_dim);
    EXPECT_EQ(lambda_got.cols(), this->n_design_dim);

    // Check contents of `beta_draws_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(std::abs(lambda_got(i, j)), std::abs(lowrank_lambda_expected[i*this->n_design_dim + j]), 1e-7);
	}
    }
}

TEST_F(CppRateResTest, sherman_r) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);
    const Eigen::MatrixXd &svd_cov_u = decompose_covariance_matrix(cov);
    const Eigen::MatrixXd &lambda = create_lambda(svd_cov_u);
    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);

    const Eigen::MatrixXd &sherman_r_got = sherman_r(lambda, cov.col(1), cov.col(1));

    // Check dimensions of `sherman_r_got`
    EXPECT_EQ(sherman_r_got.rows(), this->n_design_dim);
    EXPECT_EQ(sherman_r_got.cols(), this->n_design_dim);

    // Check contents of `sherman_r_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(sherman_r_got(i, j), this->sherman_r_expected[i*this->n_design_dim + j], 1e-6);
	}
    }
}

TEST_F(CppRateResTest, dropped_predictor_kld) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);
    const Eigen::MatrixXd &svd_cov_u = decompose_covariance_matrix(cov);
    const Eigen::MatrixXd &lambda = create_lambda(svd_cov_u);
    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);

    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(dropped_predictor_kld(lambda, cov, col_means_beta, i), this->expected_KLD[i], 1e-7);
    }
}

TEST_F(CppRateResTest, rate_from_kld) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);
    const Eigen::MatrixXd &svd_cov_u = decompose_covariance_matrix(cov);
    const Eigen::MatrixXd &lambda = create_lambda(svd_cov_u);
    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);
    std::vector<double> KLD(this->n_design_dim);
    double KLD_sum = 0.0;
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	KLD[i] = dropped_predictor_kld(lambda, cov, col_means_beta, i);
	KLD_sum += KLD[i];
    }

    const std::vector<double> &RATE_got = rate_from_kld(KLD, KLD_sum);

    // Check dimensions of `RATE_got`
    EXPECT_EQ(RATE_got.size(), this->n_design_dim);

    // Check contents of `RATE_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(RATE_got[i], this->expected_RATE[i], 1e-7);
    }
}

TEST_F(CppRateResTest, delta_rate) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);
    const Eigen::MatrixXd &svd_cov_u = decompose_covariance_matrix(cov);
    const Eigen::MatrixXd &lambda = create_lambda(svd_cov_u);
    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);
    std::vector<double> KLD(this->n_design_dim);
    double KLD_sum = 0.0;
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	KLD[i] = dropped_predictor_kld(lambda, cov, col_means_beta, i);
	KLD_sum += KLD[i];
    }
    const std::vector<double> &RATE = rate_from_kld(KLD, KLD_sum);

    const double delta_got = rate_delta(RATE);
    EXPECT_NEAR(delta_got, this->expected_Delta, 1e-7);
}

TEST_F(CppRateResTest, delta_to_ess) {
    Eigen::SparseMatrix<double> mat = sparsify_design_matrix<double, bool>(this->n_obs, this->n_design_dim, this->design_matrix);
    Eigen::MatrixXd f_draws_mat(this->n_f_draws, this->n_obs);
    for (size_t i = 0; i < this->n_f_draws; ++i) {
	for (size_t j = 0; j < this->n_obs; ++j) {
	    f_draws_mat(i, j) = this->f_draws[i*this->n_obs + j];
	}
    }
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(mat, f_draws_mat);
    const Eigen::MatrixXd &cov = covariance_matrix(beta_draws);
    const Eigen::MatrixXd &svd_cov_u = decompose_covariance_matrix(cov);
    const Eigen::MatrixXd &lambda = create_lambda(svd_cov_u);
    const Eigen::VectorXd &col_means_beta = col_means(beta_draws);
    std::vector<double> KLD(this->n_design_dim);
    double KLD_sum = 0.0;
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	KLD[i] = dropped_predictor_kld(lambda, cov, col_means_beta, i);
	KLD_sum += KLD[i];
    }
    const std::vector<double> &RATE = rate_from_kld(KLD, KLD_sum);
    const double delta = rate_delta(RATE);

    const double ess_got = delta_to_ess(delta);
    EXPECT_NEAR(ess_got, this->expected_ESS, 1e-7);
}
