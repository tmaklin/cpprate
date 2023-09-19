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
#include <thread>
#include <future>
#include <functional>

#include "BS_thread_pool.hpp"

#include "CovarianceMatrix.hpp"
#include "RATE_res.hpp"

inline void decompose_design_matrix(const Eigen::SparseMatrix<double> &design_matrix, const size_t svd_rank, const double prop_var,
			     Eigen::MatrixXd *u, Eigen::MatrixXd *v) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    RedSVD::RedSVD<Eigen::MatrixXd> svd;
    svd.compute(design_matrix, svd_rank);

    std::vector<double> px(svd_rank);
    std::vector<bool> r_X(svd_rank);

    // Calculate weight of each SVD component
    double sv_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sv_sum)
    for (size_t i = 0; i < svd_rank; ++i) {
	px[i] = svd.singularValues()[i]*svd.singularValues()[i];
	sv_sum += px[i];
    }

    // Find number of SVD components required to explain `prop_var` of the variance
    size_t num_r_X_set = 0;
    for (size_t i = 0; i < svd_rank; ++i) {
	px[i] /= sv_sum;
	if (i > 0) {
	    px[i] += px[i - 1];
	}
	r_X[i] = (svd.singularValues()[i] > (double)1e-10) && (std::abs(px[i] - prop_var) > 1e-7);
	num_r_X_set += r_X[i];
    }

    // Construct a vector for containing indexes of SVD components to keep to explain `prop_var`
    Eigen::VectorXi keep_dim(num_r_X_set);
    size_t k = 0;
    for (size_t i = 0; i < svd_rank; ++i) {
	if (r_X[i]) {
	    keep_dim[k] = i;
	    ++k;
	}
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_r_X_set; ++i) {
	for (Eigen::Index j = 0; j < svd.matrixU().rows(); ++j) {
	    svd.matrixU()(j, keep_dim[i]) /= svd.singularValues()[keep_dim[i]];
	}
    }

    (*u) = std::move(svd.matrixU()(Eigen::indexing::all, keep_dim));
    u->transposeInPlace();
    (*v) = std::move(svd.matrixV()(Eigen::indexing::all, keep_dim));
}

inline Eigen::MatrixXd nonlinear_coefficients(const Eigen::SparseMatrix<double> &design_matrix, const Eigen::MatrixXd &f_draws) {
    // Calculate f_draws * pseudo_inverse(design_matrix) with a linear equations solver
    return Eigen::MatrixXd(design_matrix).completeOrthogonalDecomposition().solve(f_draws.transpose()).transpose();
}

inline Eigen::VectorXd col_means(const Eigen::MatrixXd &mat) {
    return mat.colwise().mean();
}

inline std::vector<double> col_means2(const Eigen::MatrixXd &mat) {
    const Eigen::VectorXd &col_means = mat.colwise().mean();
    std::vector<double> ret(col_means.size());
    for (size_t i = 0; i < ret.size(); ++i) {
	ret[i] = col_means(i);
    }
    return ret;
}

inline double get_alpha(const CovMat &cov_beta, const std::vector<double> &log_u, const size_t predictor_id) {
    // TODO tests

    size_t dim = log_u.size();
    std::vector<double> predictor_col(dim);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dim; ++i) {
	predictor_col[i] = cov_beta.get_U_val(log_u, i, i);
    }

    std::vector<double> alpha_parts(dim, 0.0);
    double alpha_parts_max = 0.0;
#pragma omp parallel for schedule(guided) reduction(vec_double_plus:alpha_parts) reduction(max:alpha_parts_max)
    for (int64_t j = dim - 1; j >= 0; --j) {
	std::vector<double> res_vec(dim, 0.0);
	if ((size_t)j != predictor_id) {
	    double max_elem = 0.0;

	    res_vec[predictor_id] += predictor_col[j] + predictor_col[j] + predictor_col[j];
	    max_elem = (max_elem > res_vec[predictor_id] ? max_elem : res_vec[predictor_id]);

	    for (size_t i = (j + 1); i < dim; ++i) {
		if (i != predictor_id) {
		    res_vec[i] += predictor_col[i] + cov_beta.get_U_val(log_u, j, i) + predictor_col[j];
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
	    alpha_parts_max = (alpha_parts_max > alpha_parts[j] ? alpha_parts_max : alpha_parts[j]);
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

inline std::vector<double> approximate_beta_means(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &u, const Eigen::MatrixXd &v) {
    const Eigen::VectorXd &col_means_f = col_means(f_draws);
    const Eigen::VectorXd &approximate_col_means = v*u*col_means_f;
    std::vector<double> ret(approximate_col_means.size(), 0.0);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < approximate_col_means.size(); ++i) {
	ret[i] = approximate_col_means(i);
    }
    return ret;
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

inline std::vector<double> run_RATE(const std::vector<double> &col_means_beta, const CovMat &cov_beta,
		      const std::vector<size_t> &ids_to_test, const size_t id_start, const size_t id_end, const size_t n_snps) {
    std::vector<double> log_KLD(n_snps, -36.84136); // log(1e-16) = -36.84136

    bool test_in_order = ids_to_test.size() == 0;
    size_t start = test_in_order ? id_start : 0;
    size_t end = test_in_order ? id_end : ids_to_test.size();
    for (size_t i = start; i < end; ++i) {
	const size_t snp_id = test_in_order ? i : ids_to_test[i];
	const std::vector<double> &cov_beta_col = cov_beta.get_col(snp_id);
	const double log_alpha = get_alpha(cov_beta, cov_beta_col, snp_id);
	const double log_m = std::log(std::abs(col_means_beta[snp_id]) + 1e-16);
	log_KLD[snp_id] = log_m + log_m + log_alpha + std::log(0.5);
    }
    return log_KLD;
}

#endif
