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

struct RATEd {
public:
    double ESS;
    double Delta;
    std::vector<double> RATE;
    std::vector<double> KLD;

    RATEd(double _ESS, double _Delta, std::vector<double> _RATE, std::vector<double> _KLD) {
	this->ESS = _ESS;
	this->Delta = _Delta;
	this->RATE = _RATE;
	this->KLD = _KLD;
    }
};

template <typename T, typename V>
Eigen::SparseMatrix<T> sparsify_design_matrix(const size_t n_rows, const size_t n_cols, const std::vector<V> &vals) {
    std::vector<Eigen::Triplet<T>> triplet_list;
    triplet_list.reserve(n_rows*n_cols);

    for (size_t i = 0; i < n_rows; ++i) {
	for (size_t j = 0; j < n_cols; ++j) {
	    if (vals[i*n_cols + j] != (V)0) {
		T val = (T)vals[i*n_cols + j];
		triplet_list.emplace_back(Eigen::Triplet<T>(i, j, val));
	    }
	}
    }
    Eigen::SparseMatrix<T> mat(n_rows, n_cols);
    mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return mat;
}

inline Eigen::MatrixXd sherman_r(const Eigen::MatrixXd &ap, const Eigen::VectorXd &u, const Eigen::VectorXd &v) {
    Eigen::MatrixXd nominator = ap * u * Eigen::Transpose<const Eigen::VectorXd>(v) * ap;
    Eigen::MatrixXd denominator = Eigen::MatrixXd::Ones(ap.rows(), ap.cols()) + u * Eigen::Transpose<const Eigen::VectorXd>(v) * ap;
    return (ap - (nominator.array()/denominator.array()).matrix());
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
    const Eigen::MatrixXd &beta_draws = f_draws*inv_X.adjoint(); // TODO just fill in the transpose.
    return beta_draws;

}

inline Eigen::MatrixXd covariance_matrix(const Eigen::MatrixXd &in) {
    const Eigen::MatrixXd &centered = in.rowwise() - in.colwise().mean();
    return (centered.adjoint() * centered) / double(in.rows() - 1);
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

inline Eigen::MatrixXd decompose_covariance_approximation(const Eigen::MatrixXd &covariance_matrix, const Eigen::MatrixXd &v, const size_t svd_rank) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    Eigen::VectorXd svd_singular_values;
    Eigen::MatrixXd svd_U;
    Eigen::MatrixXd svd_V;

    svd<Eigen::MatrixXd>(covariance_matrix, svd_rank, &svd_U, &svd_V, &svd_singular_values);

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

    // We could probably reuse the svd_U or svd_V matrix ?
    return (U*inv_v).adjoint(); // TODO just fill in the transpose of U (its also faster to traverse that way)
}

inline Eigen::VectorXd col_means(const Eigen::MatrixXd &mat) {
    return mat.colwise().mean();
}

inline Eigen::MatrixXd create_lambda(const Eigen::MatrixXd &U) {
    return U.adjoint()*U;
}

inline double dropped_predictor_kld(const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &cov_beta, const Eigen::VectorXd &mean_beta, const size_t predictor_id) {
    double m = std::abs(mean_beta[predictor_id]);
    Eigen::MatrixXd U_Lambda_sub = sherman_r(lambda, cov_beta.col(predictor_id), cov_beta.col(predictor_id));
    Eigen::VectorXi dropped_predictor(mean_beta.size() - 1);
    size_t k = 0;
    for (size_t j = 0; j < mean_beta.size(); ++j) {
	if (j != predictor_id) {
	    dropped_predictor[k] = j;
	    ++k;
	}
    }
    Eigen::MatrixXd alpha = U_Lambda_sub(dropped_predictor, predictor_id).adjoint()*U_Lambda_sub(dropped_predictor, dropped_predictor)*U_Lambda_sub(dropped_predictor, predictor_id);
    return 0.5*m*alpha(0, 0)*m;
}

inline std::vector<double> rate_from_kld(const std::vector<double> &kld, const double kld_sum) {
    std::vector<double> RATE(kld.size());
    for (size_t i = 0; i < kld.size(); ++i) {
	RATE[i] = kld[i]/kld_sum;
    }
    return RATE;
}

inline double rate_delta(const std::vector<double> &RATE) {
    double Delta = 0.0;
    for (size_t i = 0; i < RATE.size(); ++i) {
	Delta += RATE[i]*std::log(RATE.size()*(RATE[i] + 1e-16));
    }
    return Delta;
}

inline double delta_to_ess(const double delta) {
    return 1.0/(1.0 + delta)*100.0;
}

inline Eigen::MatrixXd project_f_draws(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &v) {
    const Eigen::MatrixXd &cov_f_draws = covariance_matrix(f_draws);
    return v*cov_f_draws*v.adjoint();
}

inline Eigen::MatrixXd approximate_cov_beta(const Eigen::MatrixXd &project_f_draws, const Eigen::MatrixXd &v) {
    return v*project_f_draws*v.adjoint();
}

inline Eigen::VectorXd approximate_beta_means(const Eigen::MatrixXd &f_draws, const Eigen::MatrixXd &u, const Eigen::MatrixXd &v) {
    return v*u*col_means(f_draws);
}

inline RATEd RATE(const size_t n_obs, const size_t n_snps, const size_t n_f_draws, std::vector<bool> &design_matrix, const std::vector<double> &f_draws, const bool low_rank=false, const size_t low_rank_rank=0) {
    // ## WARNING: Do not compile with -ffast-math

    size_t svd_rank = std::min(n_obs, n_snps);
    const double prop_var = 1.0; // TODO email the authors if this is right (if prop.var == 1.0 the last component is always ignored)?

    const Eigen::SparseMatrix<double> &X = sparsify_design_matrix<double, bool>(n_obs, n_snps, design_matrix);
    Eigen::MatrixXd f_draws_mat(n_f_draws, n_obs);
    for (size_t i = 0; i < n_f_draws; ++i) {
	for (size_t j = 0; j < n_obs; ++j) {
	    f_draws_mat(i, j) = f_draws[i*n_obs + j];
	}
    }

    Eigen::MatrixXd cov_beta;
    Eigen::VectorXd col_means_beta;
    Eigen::MatrixXd Lambda;
    if (low_rank) {
	Eigen::MatrixXd u;
	Eigen::MatrixXd v;
	decompose_design_matrix(X, svd_rank, prop_var, &u, &v);
	const Eigen::MatrixXd &Sigma_star = project_f_draws(f_draws_mat, u);
	const Eigen::MatrixXd &svd_cov_beta_u = decompose_covariance_approximation(Sigma_star, v, low_rank_rank).adjoint(); // This does not work
	cov_beta = approximate_cov_beta(Sigma_star, v);
	col_means_beta = approximate_beta_means(f_draws_mat, u, v);
	Lambda = create_lambda(svd_cov_beta_u);
    } else {
	const Eigen::MatrixXd &beta_draws = nonlinear_coefficients(X, f_draws_mat);
	cov_beta = covariance_matrix(beta_draws);
	const Eigen::MatrixXd &svd_cov_beta_u = decompose_covariance_matrix(cov_beta);
	col_means_beta = col_means(beta_draws);
	Lambda = create_lambda(svd_cov_beta_u);
    }

    std::vector<double> KLD(n_snps);

    double KLD_sum = 0.0;
    for (size_t i = 0; i < n_snps; ++i) {
	KLD[i] = dropped_predictor_kld(Lambda, cov_beta, col_means_beta, i);
	KLD_sum += KLD[i];
    }

    const std::vector<double> &RATE = rate_from_kld(KLD, KLD_sum);
    const double Delta = rate_delta(RATE);

    const double ESS = delta_to_ess(Delta);

    return RATEd(ESS, Delta, RATE, KLD);
}

#endif
