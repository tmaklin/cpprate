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

#include "cpprate_blas_config.hpp"
#include "cpprate_openmp_config.hpp"

#include <Eigen/SparseCore>

#include "RedSVD.h"

#include <vector>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <numeric>

inline std::vector<double> rate_from_kld(const std::vector<double> &log_kld, const double kld_sum) {
    std::vector<double> RATE(log_kld.size());
    double log_kld_sum = std::log(kld_sum);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < log_kld.size(); ++i) {
	RATE[i] = std::exp(log_kld[i] - log_kld_sum);
    }
    return RATE;
}

inline std::vector<double> rate_from_log_kld(const std::vector<double> &log_kld) {
    double max_elem = 0.0;
    // TODO pragma with custom reduction to find maximum
    for (size_t i = 0; i < log_kld.size(); ++i) {
	max_elem = (max_elem > log_kld[i] ? max_elem : log_kld[i]);
    }
    double tmp_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:tmp_sum)
    for (size_t i = 0; i < log_kld.size(); ++i) {
	tmp_sum += std::exp(log_kld[i] - max_elem);
    }
    double log_kld_sum = std::log(tmp_sum) + max_elem;

    std::vector<double> RATE(log_kld.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < log_kld.size(); ++i) {
	RATE[i] = std::exp(log_kld[i] - log_kld_sum);
    }
    return RATE;
}

inline double rate_delta(const std::vector<double> &RATE) {
    double Delta = 0.0;
    size_t num_snps = RATE.size();
#pragma omp parallel for schedule(static) reduction(+:Delta)
    for (size_t i = 0; i < num_snps; ++i) {
	Delta += RATE[i]*(std::log(num_snps) + std::log(RATE[i] + 1e-16));
    }

    return Delta;
}

inline double delta_to_ess(const double delta) {
    return std::exp(std::log(1.0) - std::log(1.0 + delta))*100.0;
}

struct RATEd {
public:
    double ESS;
    double Delta;
    std::vector<double> RATE;
    std::vector<double> KLD;

    RATEd() = default;

    RATEd(double _ESS, double _Delta, std::vector<double> _RATE, std::vector<double> _KLD) {
	this->ESS = _ESS;
	this->Delta = _Delta;
	this->RATE = _RATE;
	this->KLD = _KLD;
    }

    RATEd(std::vector<double> _log_KLD) {
	std::transform(_log_KLD.begin(), _log_KLD.end(), std::back_inserter(this->KLD), static_cast<double(*)(double)>(std::exp));
	this->RATE = rate_from_log_kld(_log_KLD);
	this->Delta = rate_delta(RATE);
	this->ESS = delta_to_ess(Delta);
    }

    RATEd(std::vector<double> _KLD, std::vector<double> _RATE) {
	this->KLD = _KLD;
	this->RATE = _RATE;
	this->Delta = rate_delta(RATE);
	this->ESS = delta_to_ess(Delta);
    }
};

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

inline std::vector<double> sherman_r(const std::vector<double> &abs_flat_lambda, const std::vector<double> &u) {
    size_t dim = u.size();
    std::vector<double> tmp(dim * (dim + 1)/2, 0.0);

#pragma omp parallel for schedule(guided) // Last chunks are very small so "reverse guided" works ok
    for (int64_t j = dim - 1; j >= 0; --j) {
	size_t col_start = j * dim - j * (j - 1)/2 - j;
	for (size_t i = j; i < dim; ++i) {
	    double log_outer_prod = std::log(std::abs(u[i]) + 1e-16) + std::log(std::abs(u[j]) + 1e-16);
	    double log_flat_lambda = std::log(abs_flat_lambda[col_start + 1]);
	    double log_val = log_outer_prod + log_flat_lambda;
	    tmp[col_start + i] = abs_flat_lambda[col_start + i] - std::exp(log_flat_lambda + log_val - std::log1p(std::exp(log_val)));
	}
    }

    return tmp;
}

inline double create_denominator(const Eigen::MatrixXd &log_v_Sigma_star, const Eigen::VectorXd &svd_v_col) {
    // TODO: tests
    std::vector<double> square_norms(log_v_Sigma_star.cols(), 0.0);

    double max_element = 0.0;
#pragma omp parallel
    {
	double local_max = 0.0;
#pragma omp for schedule(static) reduction(vec_double_plus:square_norms)
	for (size_t j = 0; j < log_v_Sigma_star.cols(); ++j) {
	    for (size_t i = 0; i < log_v_Sigma_star.rows(); ++i) {
		double log_prod = log_v_Sigma_star(i, j) + std::log(std::abs(svd_v_col(j) + 1e-16));
		square_norms[j] += log_prod + log_prod;
	    }
	    local_max = (local_max > square_norms[j] ? local_max : square_norms[j]);
	}

#pragma omp critical
	{
	    if (local_max > max_element) {
		max_element = local_max;
	    }
	}
    }


    double logsumexp = 0.0;

#pragma omp parallel for schedule(static) reduction(+:logsumexp)
    for (size_t j = 0; j < log_v_Sigma_star.cols(); ++j) {
	logsumexp += std::exp(square_norms[j] - max_element);
    }

    logsumexp += std::log(logsumexp) + max_element;

    return std::exp(logsumexp);
}

inline std::vector<double> create_log_nominator(const Eigen::MatrixXd &log_f_Lambda, const Eigen::VectorXd &svd_v_col) {
    // TODO: tests
    size_t dim = log_f_Lambda.rows();
    std::vector<double> tmp(dim, 0.0);

#pragma omp parallel for schedule(static) reduction(vec_double_plus:tmp)
    for (size_t j = 0; j < log_f_Lambda.cols(); ++j) {
	for (size_t i = 0; i < dim; ++i) {
	    tmp[i] += log_f_Lambda(i, j) + std::log(std::abs(svd_v_col(j)) + 1e-16);
	}
    }

    std::vector<double> log_nominator(dim * (dim + 1)/2, 0.0);

#pragma omp parallel for schedule(guided) // Last chunks are very small so "reverse guided" works ok
    for (int64_t j = dim - 1; j >= 0; --j) {
	size_t col_start = j * dim - j * (j - 1)/2 - j;
	for (size_t i = j; i < dim; ++i) {
	    log_nominator[col_start + i] = tmp[i] + tmp[j];
	}
    }

    return log_nominator;
}

inline std::vector<double> sherman_r_lowrank(const std::vector<double> &flat_Lambda, const Eigen::MatrixXd &log_f_Lambda, const Eigen::MatrixXd &log_v_Sigma_star, const Eigen::VectorXd &svd_v_col) {
    // TODO: tests
    const double log_denominator = std::log1p(create_denominator(log_v_Sigma_star, svd_v_col));
    std::vector<double> log_nominator = create_log_nominator(log_f_Lambda, svd_v_col);

    size_t dim = log_f_Lambda.rows() * (log_f_Lambda.rows() + 1)/2;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dim; ++i) {
	log_nominator[i] = flat_Lambda[i] - std::exp(log_nominator[i] - log_denominator);
    }

    return log_nominator;
}

template <typename T>
inline void svd(const T &design_matrix, const size_t svd_rank,
		Eigen::MatrixXd *u, Eigen::MatrixXd *v, Eigen::VectorXd *d) {
    RedSVD::RedSVD<T> mat(design_matrix, svd_rank);
    (*u) = std::move(mat.matrixU());
    (*v) = std::move(mat.matrixV());
    (*d) = std::move(mat.singularValues());
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

inline Eigen::MatrixXd nonlinear_coefficients(const Eigen::SparseMatrix<double> &design_matrix, const Eigen::MatrixXd &f_draws) {
    const Eigen::MatrixXd &inv_X = Eigen::MatrixXd(design_matrix).completeOrthogonalDecomposition().pseudoInverse();
    return f_draws*inv_X.transpose();

}

inline Eigen::MatrixXd covariance_matrix(const Eigen::MatrixXd &in) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(in.cols(), in.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate((in.rowwise() - in.colwise().mean()).transpose());
    tmp.array() /= double(in.rows() - 1);
    tmp.template triangularView<Eigen::Upper>() = tmp.transpose();
    return tmp;
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

inline Eigen::VectorXd col_means(const Eigen::MatrixXd &mat) {
    return mat.colwise().mean();
}

inline Eigen::MatrixXd create_lambda(const Eigen::MatrixXd &U) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(U.cols(), U.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(U.transpose());
    tmp.template triangularView<Eigen::Upper>() = tmp.transpose();
    return tmp;
}

inline double get_alpha(const std::vector<double> &U_Lambda_sub_flat, const size_t dim, const size_t predictor_id) {
    // TODO tests
    double alpha = 0.0;
    std::vector<double> predictor_col = std::move(get_col(U_Lambda_sub_flat, dim, dim, predictor_id));
    for (size_t i = 0; i < predictor_col.size(); ++i) {
	predictor_col[i] = std::log(std::abs(predictor_col[i]) + 1e-16);
    }
    std::vector<double> alpha_parts(dim, 0.0);
    double alpha_parts_max = 0.0;
#pragma omp parallel
    {
	double local_max = 0.0;
#pragma omp for schedule(guided) reduction(vec_double_plus:alpha_parts)
	for (int64_t j = dim - 1; j >= 0; --j) {
	    std::vector<double> res_vec(dim, 0.0);
	    if (j != predictor_id) {
		double max_elem = 0.0;

		size_t col_start = j * dim - j * (j - 1)/2 - j;
		res_vec[predictor_id] += predictor_col[j] + std::log(std::abs(U_Lambda_sub_flat[col_start + j]) + 1e-16) + predictor_col[j];
		max_elem = (max_elem > res_vec[predictor_id] ? max_elem : res_vec[predictor_id]);
		for (size_t i = (j + 1); i < dim; ++i) {
		    if (i != predictor_id) {
			res_vec[i] += predictor_col[i] + std::log(std::abs(U_Lambda_sub_flat[col_start + i]) + 1e-16) + predictor_col[j];
		    }
		    max_elem = (max_elem > res_vec[i] ? max_elem : res_vec[i]);
		}

		double tmp_sum = 0.0;
		for (size_t i = 0; i < dim; ++i) {
		    double val = std::exp(res_vec[i] - max_elem);
		    tmp_sum += val;
		    if (i != predictor_id) {
			tmp_sum += val;
		    }
		}
		alpha_parts[j] += std::log(tmp_sum) + max_elem;
		local_max = (local_max > alpha_parts[j] ? local_max : alpha_parts[j]);
#pragma omp critical
		{
		    if (local_max > alpha_parts_max) {
			alpha_parts_max = local_max;
		    }
		}
	    }
	}
    }

    double alpha_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:alpha_sum)
    for (size_t i = 0; i < dim; ++i) {
	if (i != predictor_id) {
	    alpha_sum += std::exp(alpha_parts[i] - alpha_parts_max);
	}
    }

    double log_alpha = std::log(alpha_sum) + alpha_parts_max;
    return log_alpha;
}

inline double dropped_predictor_kld(const std::vector<double> &flat_lambda, const std::vector<double> &cov_beta_col, const double mean_beta, const size_t predictor_id) {
    double log_m = std::log(std::abs(mean_beta) + 1e-16); // Sign is lost in the return value so doesn't matter
    const std::vector<double> &U_Lambda_sub_flat = sherman_r(flat_lambda, cov_beta_col);

    size_t dim = cov_beta_col.size();
    const double log_alpha = get_alpha(U_Lambda_sub_flat, dim, predictor_id);

    return std::log(0.5) + log_m + log_alpha + log_m;
}

inline double dropped_predictor_kld_lowrank(const std::vector<double> &flat_Lambda, const Eigen::MatrixXd &log_f_Lambda, const Eigen::MatrixXd &log_v_Sigma_star, const Eigen::VectorXd &svd_v_col, const double mean_beta, const size_t predictor_id) {
    // TODO: tests
    const std::vector<double> &flat_U_Lambda_sub = sherman_r_lowrank(flat_Lambda, log_f_Lambda, log_v_Sigma_star, svd_v_col);

    size_t dim = log_f_Lambda.rows();
    const double log_alpha = get_alpha(flat_U_Lambda_sub, dim, predictor_id);

    double log_m = std::log(std::abs(mean_beta) + 1e-16);
    return std::log(0.5) + log_m + log_alpha + log_m;
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

inline std::vector<double> flatten_triangular(const Eigen::MatrixXd &triangular) {
    // TODO tests
    // Flatten a lower triangular matrix `triangular`
    // Note: assumes that `triangular` is rectangular.
    // Returns:
    //   flattened: vector containing values at or below the diagonal
    //              from `triangular` column-wise from left to right.
    //
    size_t dim = triangular.rows();
    std::vector<double> flattened(dim * (dim + 1)/2, 0.0);

#pragma omp parallel for schedule(guided)
    for (int64_t j = dim - 1; j >= 0; --j) {
	size_t col_start = j * dim - j * (j - 1)/2 - j;
	for (size_t i = j; i < dim; ++i) {
	    flattened[col_start + i] = triangular(i, j);
	}
    }

    return flattened;
}

inline RATEd RATE_lowrank(Eigen::MatrixXd &f_draws, Eigen::SparseMatrix<double> &design_matrix, const std::vector<size_t> &ids_to_test, const size_t id_start, const size_t id_end, const size_t n_snps, const size_t svd_rank, const double prop_var) {
    // ## WARNING: Do not compile with -ffast-math

    Eigen::MatrixXd u;
    Eigen::MatrixXd svd_design_matrix_v;
    decompose_design_matrix(design_matrix, svd_rank, prop_var, &u, &svd_design_matrix_v);
    design_matrix.resize(0, 0);

    const Eigen::VectorXd &col_means_beta = approximate_beta_means(f_draws, u, svd_design_matrix_v);
    Eigen::MatrixXd v_Sigma_star = std::move(svd_design_matrix_v*project_f_draws(f_draws, u).triangularView<Eigen::Lower>());

    Eigen::MatrixXd Lambda_f;
    std::vector<double> flat_Lambda;

    {
	Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(n_snps, n_snps);
	Eigen::MatrixXd Lambda_chol = decompose_covariance_approximation(project_f_draws(f_draws, u), svd_design_matrix_v, svd_rank);
	f_draws.resize(0, 0);
	Lambda.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda_chol);
	Lambda_f = Lambda.triangularView<Eigen::Lower>() * v_Sigma_star;
	flat_Lambda = flatten_triangular(Lambda);
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < v_Sigma_star.cols(); ++i) {
	    for (size_t j = 0; j < v_Sigma_star.rows(); ++j) {
		v_Sigma_star(j, i) = std::log(std::abs(v_Sigma_star(j, i)) + 1e-16) + std::log(std::abs(Lambda_chol(j, i)) + 1e-16);
	    }
	}
    }

    {
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < Lambda_f.cols(); ++i) {
	    for (size_t j = 0; j < Lambda_f.rows(); ++j) {
		Lambda_f(j, i) = std::log(std::abs(Lambda_f(j, i)) + 1e-16);
	    }
	}
    }

    u.resize(0, 0);

    svd_design_matrix_v.transposeInPlace();

    std::vector<double> log_KLD(n_snps, -36.84136); // log(1e-16) = -36.84136

    if (ids_to_test.size() == 0) {
#pragma omp parallel for schedule(static)
	for (size_t i = id_start; i < id_end; ++i) {
	    log_KLD[i] = dropped_predictor_kld_lowrank(flat_Lambda, Lambda_f, v_Sigma_star, svd_design_matrix_v.col(i), col_means_beta[i], i);
	}
    } else {
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < ids_to_test.size(); ++i) {
	    size_t snp_id = ids_to_test[i] - 1;
	    log_KLD[snp_id] = dropped_predictor_kld_lowrank(flat_Lambda, Lambda_f, v_Sigma_star, svd_design_matrix_v.col(snp_id), col_means_beta[snp_id], snp_id);
	}
    }

    return RATEd(log_KLD);
}

inline RATEd RATE_beta_draws(const Eigen::MatrixXd &beta_draws, const std::vector<size_t> &ids_to_test, const size_t id_start, const size_t id_end, const size_t n_snps) {
    // ## WARNING: Do not compile with -ffast-math

    std::vector<double> flat_lambda;
    Eigen::VectorXd col_means_beta;
    std::vector<double> flat_cov_beta;
    size_t dim = beta_draws.cols();

    {
	col_means_beta = col_means(beta_draws);
	const Eigen::MatrixXd &cov_beta = covariance_matrix(beta_draws);
	flat_lambda = flatten_triangular(create_lambda(decompose_covariance_matrix(cov_beta)));
	flat_cov_beta = flatten_triangular(cov_beta);
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < flat_lambda.size(); ++i) {
	    flat_lambda[i] = std::abs(flat_lambda[i]) + 1e-16;
	}
    }

    std::vector<double> log_KLD(n_snps, 0.0);

    if (ids_to_test.size() == 0) {
#pragma omp parallel for schedule(static)
	for (size_t i = id_start; i < id_end; ++i) {
	    const std::vector<double> &cov_beta_col = get_col(flat_cov_beta, dim, dim, i);
	    log_KLD[i] = dropped_predictor_kld(flat_lambda, cov_beta_col, col_means_beta[i], i);
	}
    } else {
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < ids_to_test.size(); ++i) {
	    const std::vector<double> &cov_beta_col = get_col(flat_cov_beta, dim, dim, i);
	    size_t snp_id = ids_to_test[i] - 1;
	    log_KLD[snp_id] = dropped_predictor_kld(flat_lambda, cov_beta_col, col_means_beta[snp_id], snp_id);
	}
    }

    return RATEd(log_KLD);
}

inline RATEd RATE_fullrank(const Eigen::MatrixXd &f_draws, const Eigen::SparseMatrix<double> &design_matrix, const std::vector<size_t> &ids_to_test, const size_t id_start, const size_t id_end, const size_t n_snps) {
    // ## WARNING: Do not compile with -ffast-math
    const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(design_matrix, f_draws);

    const RATEd &res = RATE_beta_draws(beta_draws, ids_to_test, id_start, id_end, n_snps);

    return res;
}

#endif
