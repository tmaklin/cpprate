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
#ifndef CPPRATE_LOWRANK_HPP
#define CPPRATE_LOWRANK_HPP

#include <cstddef>
#include <vector>
#include <cmath>

#include "cpprate_openmp_config.hpp"
#include "cpprate_blas_config.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "flat_triangular.hpp"
#include "svd_wrapper.hpp"
#include "RATE_res.hpp"
#include "util.hpp"

inline double create_denominator(const Eigen::MatrixXd &t_Lambda_chol, const Eigen::MatrixXd &v_Sigma_star, const Eigen::VectorXd &svd_v_col) {
    // TODO: tests
    double square_norm = 0.0;
#pragma omp parallel for schedule(static) reduction(+:square_norm)
    for (size_t j = 0; j < v_Sigma_star.cols(); ++j) {
	for (size_t i = 0; i < v_Sigma_star.rows(); ++i) {
	    double prod = (v_Sigma_star(i, j) * svd_v_col(j)) * t_Lambda_chol(i, j);
	    square_norm += prod * prod;
	}
    }

    return square_norm + 1.0;
}

inline std::vector<double> create_nominator(const Eigen::MatrixXd &f_Lambda, const Eigen::VectorXd &svd_v_col) {
    // TODO: tests
    size_t dim = f_Lambda.rows();
    std::vector<double> tmp(dim, 0.0);

#pragma omp parallel for schedule(static) reduction(vec_double_plus:tmp)
    for (size_t j = 0; j < f_Lambda.cols(); ++j) {
	for (size_t i = 0; i < dim; ++i) {
	    tmp[i] += f_Lambda(i, j) * svd_v_col(j);
	}
    }

    std::vector<double> nominator(dim * (dim + 1)/2, 0.0);

#pragma omp parallel for schedule(guided) // Last chunks are very small so "reverse guided" works ok
    for (int64_t j = dim - 1; j >= 0; --j) {
	size_t col_start = j * dim - j * (j - 1)/2 - j;
	for (size_t i = j; i < dim; ++i) {
	    nominator[col_start + i] = tmp[i] * tmp[j];
	}
    }

    return nominator;
}

inline std::vector<double> sherman_r_lowrank(const std::vector<double> &flat_Lambda, const Eigen::MatrixXd &f_Lambda, const Eigen::MatrixXd &Lambda_chol, const Eigen::MatrixXd &v_Sigma_star, const Eigen::VectorXd &svd_v_col) {
    // TODO: tests
    const double denominator = create_denominator(Lambda_chol, v_Sigma_star, svd_v_col);
    std::vector<double> nominator = create_nominator(f_Lambda, svd_v_col);

    size_t dim = f_Lambda.rows() * (f_Lambda.rows() + 1)/2;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dim; ++i) {
	nominator[i] = flat_Lambda[i] - (nominator[i]/denominator);
    }

    return nominator;
}

inline double dropped_predictor_kld_lowrank(const std::vector<double> &flat_Lambda, const Eigen::MatrixXd &f_Lambda, const Eigen::MatrixXd &Lambda_chol, const Eigen::MatrixXd &v_Sigma_star, const Eigen::VectorXd &svd_v_col, const double mean_beta, const size_t predictor_id) {
    // TODO: tests
    const std::vector<double> &flat_U_Lambda_sub = sherman_r_lowrank(flat_Lambda, f_Lambda, Lambda_chol, v_Sigma_star, svd_v_col);

    size_t dim = f_Lambda.rows();
    const std::vector<double> &predictor_col = get_col(flat_U_Lambda_sub, dim, dim, predictor_id);
    double alpha = 0.0;

#pragma omp parallel for schedule(guided) reduction(+:alpha)
    for (int64_t j = dim - 1; j >= 0; --j) {
	if (j != predictor_id) {
	    size_t col_start = j * dim - j * (j - 1)/2 - j;
	    alpha += (predictor_col[j] * flat_U_Lambda_sub[col_start + j]) * predictor_col[j];
	    for (size_t i = (j + 1); i < dim; ++i) {
		if (i != predictor_id) {
		    double res = (predictor_col[i] * flat_U_Lambda_sub[col_start + i]) * predictor_col[j];
		    alpha += res;
		    alpha += res;
		}
	    }
	}
    }

    double log_m = std::log(std::abs(mean_beta));
    return std::log(0.5) + log_m + std::log(alpha) + log_m;
}

inline Eigen::MatrixXd decompose_covariance_approximation(const Eigen::MatrixXd &dense_covariance_matrix, const Eigen::MatrixXd &v, const size_t svd_rank) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    Eigen::VectorXd svd_singular_values;
    Eigen::MatrixXd svd_U;

    {
	Eigen::MatrixXd svd_V;
	svd<Eigen::MatrixXd>(dense_covariance_matrix, svd_rank, &svd_U, &svd_V, &svd_singular_values);
    }

    size_t dim_svd_res = svd_singular_values.size();
    std::vector<bool> r_D(dim_svd_res);

    size_t num_r_D_set = 0;
    for (size_t i = 0; i < dim_svd_res; ++i) {
	r_D[i] = svd_singular_values[i] > 1e-10;
	num_r_D_set += r_D[i];
    }

    size_t n_rows_U = svd_U.rows();
    Eigen::MatrixXd U(num_r_D_set, n_rows_U);

    size_t k = 0;
    for (size_t i = 0; i < dim_svd_res; ++i) {
	if (r_D[i]) {
	    for (size_t j = 0; j < n_rows_U; ++j) {
		double leftside = std::log(1.0) - std::log(std::sqrt(svd_singular_values[i]));
		bool sign = (leftside > 0 && svd_U(j, i) > 0);
		U(k, j) = (sign == 1 ? std::exp(leftside + std::log(std::abs(svd_U(j, i)))) : -std::exp(leftside + std::log(std::abs(svd_U(j, i)))));
	    }
	    ++k;
	}
    }
    const Eigen::MatrixXd &inv_v = v.completeOrthogonalDecomposition().pseudoInverse();

    return (U*inv_v).transpose();
}

inline Eigen::MatrixXd project_f_draws(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &v) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(v.rows(), v.rows());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(v*(f_draws.rowwise() - f_draws.colwise().mean()).transpose());
    tmp.array() /= double(f_draws.rows() - 1);
    tmp.triangularView<Eigen::Upper>() = tmp.transpose();
    return tmp;
}

inline Eigen::MatrixXd approximate_cov_beta(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &u, const Eigen::MatrixXd &v) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(v.rows(), v.rows());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(v*u*(f_draws.rowwise() - f_draws.colwise().mean()).transpose());
    tmp.array() /= double(f_draws.rows() - 1);
    return tmp.sparseView();
}

inline Eigen::VectorXd approximate_beta_means(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &u, const Eigen::MatrixXd &v) {
    return v*u*col_means(f_draws);
}

inline void decompose_design_matrix(const Eigen::SparseMatrix<double> &design_matrix, const size_t svd_rank, const double prop_var,
			     Eigen::MatrixXd *u, Eigen::MatrixXd *v) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    Eigen::VectorXd svd_X_singular_values;
    Eigen::MatrixXd svd_X_U;
    Eigen::MatrixXd svd_X_V;
    svd<Eigen::SparseMatrix<double>>(design_matrix, svd_rank, &svd_X_U, &svd_X_V, &svd_X_singular_values);

    std::vector<double> px(svd_rank);
    std::vector<bool> r_X(svd_rank);

    double sv_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sv_sum)
    for (size_t i = 0; i < svd_rank; ++i) {
	sv_sum += svd_X_singular_values[i]*svd_X_singular_values[i];
    }

    size_t num_r_X_set = 0;
    for (size_t i = 0; i < svd_rank; ++i) {
	px[i] = svd_X_singular_values[i]*svd_X_singular_values[i];
	px[i] /= sv_sum;
	if (i > 0) {
	    px[i] += px[i - 1];
	}
	r_X[i] = (svd_X_singular_values[i] > (double)1e-10) && (std::abs(px[i] - prop_var) > 1e-7);
	num_r_X_set += r_X[i];
    }

    Eigen::VectorXi keep_dim(num_r_X_set);
    size_t k = 0;
    for (size_t i = 0; i < svd_rank; ++i) {
	if (r_X[i]) {
	    keep_dim[k] = i;
	    ++k;
	}
    }
    
    size_t n_rows_U = svd_X_U.rows();
    size_t n_cols_U = svd_X_U.cols();
    (*u) = std::move(Eigen::MatrixXd(num_r_X_set, n_rows_U));

    k = 0;
    for (size_t i = 0; i < n_cols_U; ++i) {
	if (r_X[i]) {
	    for (size_t j = 0; j < n_rows_U; ++j) {
		(*u)(k, j) = (1.0/svd_X_singular_values[i])*svd_X_U(j, i);
	    }
	    ++k;
	}
    }
    (*v) = std::move(svd_X_V(Eigen::indexing::all, keep_dim));
}

inline RATEd RATE_lowrank(const Eigen::MatrixXd &f_draws, const Eigen::SparseMatrix<double> &design_matrix, const size_t n_snps, const size_t svd_rank, const double prop_var) {
    // ## WARNING: Do not compile with -ffast-math

    Eigen::MatrixXd u;
    Eigen::MatrixXd svd_design_matrix_v;
    decompose_design_matrix(design_matrix, svd_rank, prop_var, &u, &svd_design_matrix_v);

    const Eigen::VectorXd &col_means_beta = approximate_beta_means(f_draws, u, svd_design_matrix_v);
    const Eigen::MatrixXd &v_Sigma_star = svd_design_matrix_v*project_f_draws(f_draws, u).triangularView<Eigen::Lower>();
    Eigen::MatrixXd Lambda_chol = decompose_covariance_approximation(project_f_draws(f_draws, u), svd_design_matrix_v, svd_rank);

    Eigen::MatrixXd Lambda_f;
    std::vector<double> flat_Lambda;

    {
	Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(n_snps, n_snps);
	Lambda.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda_chol);
	Lambda_f = Lambda.triangularView<Eigen::Lower>() * v_Sigma_star;
	flat_Lambda = flatten_lambda(Lambda);
    }

    Lambda_chol.transposeInPlace();

    u.resize(0, 0);

    svd_design_matrix_v.transposeInPlace();

    std::vector<double> log_KLD(n_snps);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_snps; ++i) {
	log_KLD[i] = dropped_predictor_kld_lowrank(flat_Lambda, Lambda_f, Lambda_chol, v_Sigma_star, svd_design_matrix_v.col(i), col_means_beta[i], i);
    }

    return RATEd(log_KLD);
}

#endif
