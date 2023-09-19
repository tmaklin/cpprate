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
#ifndef CPPRATE_COVARIANCE_MATRIX_HPP
#define CPPRATE_COVARIANCE_MATRIX_HPP

#include "cpprate_blas_config.hpp"
#include "cpprate_openmp_config.hpp"

#include <vector>
#include <cstddef>
#include <cmath>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "RedSVD.h"

#include "CppRateRes.hpp"

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

    RedSVD::RedSVD<Eigen::MatrixXd> svd;
    svd.compute_U(inv_cov_mat, rank);

    std::vector<bool> r_D(rank);
    size_t num_r_D_set = 0;
#pragma omp parallel for schedule(static) reduction(+:num_r_D_set)
    for (size_t i = 0; i < rank; ++i) {
	r_D[i] = svd.singularValues()[i] > 1e-10;
	num_r_D_set += r_D[i];
    }

    size_t n_rows_D = svd.matrixU().rows();
    Eigen::MatrixXd u(num_r_D_set, n_rows_D);
    for (size_t i = 0; i < num_r_D_set; ++i) {
	for (size_t j = 0; j < n_rows_D; ++j) {
	    u(i, j) = std::sqrt(svd.singularValues()[i])*svd.matrixU()(j, i);
	}
    }

    return u;
}

inline Eigen::MatrixXd decompose_covariance_approximation(const Eigen::MatrixXd &dense_covariance_matrix, const Eigen::MatrixXd &v, const size_t svd_rank) {
    // Calculate the singular value decomposition of `design_matrix`
    // and return the submatrices of the decomposition that correspond
    // to nonzero eigenvalues AND explain `prop_var` of the total
    // variance (default: explain 100%).

    RedSVD::RedSVD<Eigen::MatrixXd> svd;
    svd.compute_U(dense_covariance_matrix, svd_rank);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < svd.matrixU().cols(); ++i) {
	for (size_t j = 0; j < svd.matrixU().rows(); ++j) {
	    double leftside = std::log(1.0) - std::log(std::sqrt(svd.singularValues()[i]));
	    bool sign = (leftside > 0 && svd.matrixU()(j, i) > 0);
	    double log_abs_U = std::log(std::abs(svd.matrixU()(j, i)) + 1e-16);
	    svd.matrixU()(j, i) = (sign == 1 ? std::exp(leftside + log_abs_U) : -std::exp(leftside + log_abs_U));
	}
    }

    // Use linear system solver to calculate U*pseudoInverse(v) (more efficient, see Eigen documentation)
    return v.completeOrthogonalDecomposition().transpose().solve(svd.matrixU());
}

inline Eigen::MatrixXd create_lambda(const Eigen::MatrixXd &U) {
    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(U.cols(), U.cols());
    tmp.template selfadjointView<Eigen::Lower>().rankUpdate(U.transpose());
    tmp.template triangularView<Eigen::Upper>() = tmp.transpose();
    return tmp;
}

inline std::vector<double> log_flatten_triangular(const Eigen::MatrixXd &triangular) {
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
	    flattened[col_start + i] = std::log(std::abs(triangular(i, j) + 1e-16));
	}
    }

    return flattened;
}

class CovMat {
public:
    virtual std::vector<double> get_col(const size_t col_id) const =0;
    virtual const std::vector<double>& get_flat_lambda() const =0;
    virtual const double get_U_val(const std::vector<double> &tmp, const size_t i, const size_t j) const =0;
};

class FullrankCovMat : public CovMat {
private:
    size_t dim;
    std::vector<double> flat_cov_mat;
    std::vector<double> flat_Lambda;

public:

    FullrankCovMat() = default;

    const std::vector<double>& get_flat_lambda() const override { return this->flat_Lambda; }
    const double get_U_val(const std::vector<double> &cov_beta_col, const size_t i, const size_t j) const override {
	size_t col_start = i * this->dim - i * (i - 1)/2 - i;
	double log_val = cov_beta_col[i] + cov_beta_col[j] + this->flat_Lambda[col_start + i];
	return this->flat_Lambda[col_start + i]/(this->flat_Lambda[col_start + i] + log_val - std::log1p(std::exp(log_val)));

    }

    void fill(const Eigen::MatrixXd &beta_draws) {
	const Eigen::MatrixXd &dense_cov_beta = covariance_matrix(beta_draws);
	this->dim =dense_cov_beta.rows();
	this->flat_cov_mat = log_flatten_triangular(dense_cov_beta);
	this->flat_Lambda = std::move(log_flatten_triangular(create_lambda(decompose_covariance_matrix(dense_cov_beta))));
    }

    std::vector<double> get_col(const size_t col_id) const override {
	std::vector<double> res(this->dim);

#pragma omp parallel for schedule(static)
	for (size_t i = col_id; i < this->dim; ++i) {
	    size_t pos_in_lower_tri = col_id * this->dim + i - col_id * (col_id - 1)/2 - col_id;
	    res[i] = this->flat_cov_mat[pos_in_lower_tri];
	}

#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < col_id; ++i) {
	    size_t pos_in_lower_tri = i * this->dim + col_id - i * (i - 1)/2 - i;
	    res[i] = this->flat_cov_mat[pos_in_lower_tri];
	}
	return res;
    }
};

class LowrankCovMat : public CovMat {
private:
    size_t dim;
    Eigen::MatrixXd log_f_Lambda;
    Eigen::MatrixXd log_v_Sigma_star;
    Eigen::MatrixXd log_svd_V;
    std::vector<double> flat_Lambda;

    const double get_U_val(const std::vector<double> &cov_beta_col, const size_t i, const size_t j) const override {
	size_t col_start = i * this->dim - i * (i - 1)/2 - i;
	return this->flat_Lambda[col_start + i]/(cov_beta_col[i] + cov_beta_col[j]);
    }

    double create_denominator(const size_t col_id) const {
	// TODO: tests
	std::vector<double> square_norms(this->log_v_Sigma_star.cols(), 0.0);

	double max_element = 0.0;
#pragma omp parallel for schedule(static) reduction(vec_double_plus:square_norms) reduction(max:max_element)
	for (size_t j = 0; j < this->log_v_Sigma_star.cols(); ++j) {
	    for (size_t i = 0; i < this->log_v_Sigma_star.rows(); ++i) {
		double log_prod = this->log_v_Sigma_star(i, j) + log_svd_V.col(col_id)[j];
		square_norms[j] += log_prod + log_prod;
	    }
	    max_element = (max_element > square_norms[j] ? max_element : square_norms[j]);
	}

	double logsumexp = 0.0;
#pragma omp parallel for schedule(static) reduction(+:logsumexp)
	for (size_t j = 0; j < this->log_v_Sigma_star.cols(); ++j) {
	    logsumexp += std::exp(square_norms[j] - max_element);
	}

	logsumexp += std::log(logsumexp) + max_element;

	return std::exp(logsumexp - 0.6931472);
    }

    std::vector<double> create_log_nominator(const size_t col_id) const {
	// TODO: tests
	size_t dim = log_f_Lambda.rows();
	std::vector<double> tmp(dim, 0.0);

#pragma omp parallel for schedule(static) reduction(vec_double_plus:tmp)
	for (size_t j = 0; j < log_f_Lambda.cols(); ++j) {
	    for (size_t i = 0; i < dim; ++i) {
		tmp[i] += log_f_Lambda(i, j) + log_svd_V.col(col_id)[j];
	    }
	}

	return tmp;
    }

public:
    const std::vector<double>& get_flat_lambda() const override { return this->flat_Lambda; }

    LowrankCovMat() = default;
    void construct(const Eigen::MatrixXd &proj_f_draws, const size_t svd_rank) {
	this->log_v_Sigma_star = std::move(this->log_svd_V*proj_f_draws.triangularView<Eigen::Lower>());
	this->dim = log_v_Sigma_star.rows();

	// log_f_lambda = Lambda_chol
	this->log_f_Lambda = std::move(decompose_covariance_approximation(proj_f_draws, this->log_svd_V, svd_rank));

    }

    void fill() {
	Eigen::MatrixXd Lambda_chol = std::move(this->log_f_Lambda);
	this->log_f_Lambda = Eigen::MatrixXd::Zero(Lambda_chol.rows(), Lambda_chol.rows());

	// Lambda = Lambda_chol * t(Lambda_chol)
	this->log_f_Lambda.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda_chol);

	this->flat_Lambda = log_flatten_triangular(this->log_f_Lambda);

	this->log_f_Lambda *= this->log_v_Sigma_star.triangularView<Eigen::Lower>();

#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < this->log_v_Sigma_star.cols(); ++i) {
	    for (size_t j = 0; j < this->log_v_Sigma_star.rows(); ++j) {
		this->log_v_Sigma_star(j, i) = std::log(std::abs(this->log_v_Sigma_star(j, i)) + 1e-16) + std::log(std::abs(Lambda_chol(j, i)) + 1e-16);
	    }
	}
    }

    void logarithmize_lambda() {
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < this->log_f_Lambda.cols(); ++i) {
	    for (size_t j = 0; j < this->log_f_Lambda.rows(); ++j) {
		this->log_f_Lambda(j, i) = std::log(std::abs(this->log_f_Lambda(j, i)) + 1e-16);
	    }
	}
    }

    void logarithmize_svd_V() {
	this->log_svd_V.transposeInPlace();
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < this->log_svd_V.cols(); ++i) {
	    for (size_t j = 0; j < this->log_svd_V.rows(); ++j) {
		this->log_svd_V(j, i) = std::log(std::abs(this->log_svd_V(j, i) + 1e-16));
	    }
	}
    }

    std::vector<double> get_col(const size_t col_id) const override {
	// TODO: tests
	const double log_denominator = std::log1p(this->create_denominator(col_id));
	std::vector<double> col = this->create_log_nominator(col_id);
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < col.size(); ++i) {
	    col[i] -= log_denominator;
	}

	return col;
    }

    Eigen::MatrixXd* get_svd_V_p() { return &this->log_svd_V; }
    const Eigen::MatrixXd& get_svd_V() const { return this->log_svd_V; }
};

#endif
