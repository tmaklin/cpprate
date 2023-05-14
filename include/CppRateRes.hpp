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
	this->RATE = rate_from_kld(_log_KLD, std::accumulate(this->KLD.begin(), this->KLD.end(), 0.0));
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

inline Eigen::MatrixXd sherman_r(const Eigen::MatrixXd &ap, const Eigen::VectorXd &u) {
    const Eigen::MatrixXd &tmp = u * u.adjoint() * ap;
    return (ap - ( (ap * tmp).array() / (1 + (tmp).array())).matrix());
}

inline Eigen::MatrixXd sherman_r_lowrank(const Eigen::MatrixXd &Lambda, const Eigen::MatrixXd &Lambda_chol, const Eigen::SparseMatrix<double> &Sigma_star, const Eigen::MatrixXd &svd_cov_beta_v, const size_t predictor_id) {
    // TODO: tests
    Eigen::MatrixXd denominator = std::move(Eigen::MatrixXd::Zero(svd_cov_beta_v.rows(), svd_cov_beta_v.rows()));
    denominator.template selfadjointView<Eigen::Lower>().rankUpdate((svd_cov_beta_v*Sigma_star.triangularView<Eigen::Lower>()*svd_cov_beta_v.adjoint().col(predictor_id))*Lambda_chol);

    Eigen::MatrixXd nominator = std::move(Eigen::MatrixXd::Zero(svd_cov_beta_v.rows(), svd_cov_beta_v.rows()));
    nominator.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda.triangularView<Eigen::Lower>()*(svd_cov_beta_v*Sigma_star.triangularView<Eigen::Lower>()*svd_cov_beta_v.adjoint().col(predictor_id)));

#pragma omp parallel for schedule(dynamic, 1)
    for (size_t j = 0; j < nominator.cols(); ++j) {
	for (size_t i = j; i < nominator.rows(); ++i) {
	    nominator(i, j) = Lambda(i, j) - (nominator(i, j)/(denominator(i, j) + 1.0));
	}
    }

    nominator.triangularView<Eigen::Upper>() = nominator.transpose();
    return nominator;
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
    return f_draws*inv_X.adjoint();

}

inline Eigen::MatrixXd covariance_matrix(const Eigen::MatrixXd &in) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(in.cols(), in.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate((in.rowwise() - in.colwise().mean()).adjoint());
    tmp.array() /= double(in.rows() - 1);
    tmp.template triangularView<Eigen::Upper>() = tmp.adjoint();
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
    Eigen::MatrixXd svd_V;

    svd<Eigen::MatrixXd>(inv_cov_mat, rank, &svd_U, &svd_V, &svd_singular_values);

    std::vector<bool> r_D(rank);
    size_t num_r_D_set = 0;
    for (size_t i = 0; i < rank; ++i) {
	r_D[i] = svd_singular_values[i] > 1e-10;
	num_r_D_set += r_D[i];
    }

    size_t n_rows_D = svd_U.rows();
    size_t n_cols_D = svd_U.cols();
    Eigen::MatrixXd u(num_r_D_set, n_rows_D);
    for (size_t i = 0; i < num_r_D_set; ++i) { // TODO email the authors if this is right?
	for (size_t j = 0; j < n_rows_D; ++j) {
	    u(i, j) = std::sqrt(svd_singular_values[i])*svd_U(j, i);
	}
    }

    return u;
}

inline Eigen::MatrixXd decompose_covariance_approximation(const Eigen::SparseMatrix<double> &covariance_matrix, const Eigen::MatrixXd &v, const size_t svd_rank) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    Eigen::VectorXd svd_singular_values;
    Eigen::MatrixXd svd_U;
    Eigen::MatrixXd svd_V;

    Eigen::MatrixXd dense = covariance_matrix;
    dense.triangularView<Eigen::Upper>() = dense.transpose();
    svd<Eigen::MatrixXd>(dense, svd_rank, &svd_U, &svd_V, &svd_singular_values);

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

    return (U*inv_v).adjoint();
}

inline Eigen::VectorXd col_means(const Eigen::MatrixXd &mat) {
    return mat.colwise().mean();
}

inline Eigen::MatrixXd create_lambda(const Eigen::MatrixXd &U) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(U.cols(), U.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(U.adjoint());
    tmp.template triangularView<Eigen::Upper>() = tmp.adjoint();
    return tmp;
}

inline double dropped_predictor_kld(const Eigen::MatrixXd &lambda, const Eigen::VectorXd &cov_beta_col, const double mean_beta, const size_t predictor_id) {
    double log_m = std::log(std::abs(mean_beta));
    const Eigen::MatrixXd &U_Lambda_sub = sherman_r(lambda, cov_beta_col);

    double alpha = 0.0;

    for (size_t k = 0; k < U_Lambda_sub.cols(); k++) {
	if (k != predictor_id) {
	    double tmp_sum = 0.0;
	    for (size_t j = 0; j < U_Lambda_sub.rows(); ++j) {
		if (j != predictor_id) {
		    tmp_sum += U_Lambda_sub(j, predictor_id) * U_Lambda_sub(j, k);
		}
	    }
	    alpha += tmp_sum * U_Lambda_sub(k, predictor_id);
	}
    }

    return std::log(0.5) + log_m + std::log(alpha) + log_m;
}

inline double dropped_predictor_kld_lowrank(const Eigen::MatrixXd &Lambda, const Eigen::MatrixXd &Lambda_chol, const Eigen::SparseMatrix<double> &Sigma_star, const Eigen::MatrixXd &svd_cov_beta_v, const double mean_beta, const size_t predictor_id) {
    // TODO: tests
    const Eigen::MatrixXd &U_Lambda_sub = sherman_r_lowrank(Lambda, Lambda_chol, Sigma_star, svd_cov_beta_v, predictor_id);

    double alpha = 0.0;
    const Eigen::VectorXd &predictor_col = U_Lambda_sub.col(predictor_id);
#pragma omp parallel for schedule(static) reduction(+:alpha)
    for (size_t k = 0; k < U_Lambda_sub.cols(); ++k) {
	if (k != predictor_id) {
	    alpha += predictor_col.dot(U_Lambda_sub.col(k)) * predictor_col(k);
	}
    }

    double log_m = std::log(std::abs(mean_beta));
    return std::log(0.5) + log_m + std::log(alpha) + log_m;
}

inline Eigen::SparseMatrix<double> project_f_draws(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &v) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(v.rows(), v.rows());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(v*(f_draws.rowwise() - f_draws.colwise().mean()).adjoint());
    tmp.array() /= double(f_draws.rows() - 1);
    return tmp.sparseView();
}

inline Eigen::MatrixXd approximate_cov_beta(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &u, const Eigen::MatrixXd &v) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(v.rows(), v.rows());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(v*u*(f_draws.rowwise() - f_draws.colwise().mean()).adjoint());
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
#pragma omp parallel for schedule(static)
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

inline RATEd RATE_lowrank(const Eigen::MatrixXd &f_draws, const Eigen::SparseMatrix<double> &design_matrix, const size_t n_snps, const size_t svd_rank, const double prop_var) {
    // ## WARNING: Do not compile with -ffast-math

    Eigen::MatrixXd u;
    Eigen::MatrixXd svd_design_matrix_v;
    decompose_design_matrix(design_matrix, svd_rank, prop_var, &u, &svd_design_matrix_v);

    Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(n_snps, n_snps);
    const Eigen::VectorXd &col_means_beta = approximate_beta_means(f_draws, u, svd_design_matrix_v);
    const Eigen::SparseMatrix<double> &Sigma_star = project_f_draws(f_draws, u);
    const Eigen::MatrixXd &Lambda_chol = decompose_covariance_approximation(Sigma_star, svd_design_matrix_v, svd_rank);
    Lambda.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda_chol);


    std::vector<double> log_KLD(n_snps);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_snps; ++i) {
	log_KLD[i] = dropped_predictor_kld_lowrank(Lambda, Lambda_chol, Sigma_star, svd_design_matrix_v, col_means_beta[i], i);
    }

    return RATEd(log_KLD);
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
