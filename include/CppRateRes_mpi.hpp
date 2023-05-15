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
#ifndef CPPRATE_CPPRATERES_MPI_HPP
#define CPPRATE_CPPRATERES_MPI_HPP

#include "CppRateRes.hpp"

#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include <omp.h>

#include "cpprate_mpi_config.hpp"

inline RATEd RATE_lowrank_mpi(Eigen::MatrixXd &f_draws, Eigen::SparseMatrix<double> &design_matrix, const size_t n_snps, const size_t svd_rank, const double prop_var) {
    // ## WARNING: Do not compile with -ffast-math

    // Setup MPI
    int rank;
    int n_tasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_tasks);

    Eigen::VectorXd col_means_beta(0);
    Eigen::MatrixXd v_Sigma_star(0, 0);
    Eigen::MatrixXd Lambda_chol(0, 0);
    Eigen::MatrixXd Lambda_f(0, 0);
    Eigen::MatrixXd svd_design_matrix_v(0, 0);
    Eigen::VectorXd flat_Lambda(0);
    if (rank == 0) {
	Eigen::MatrixXd u;
	decompose_design_matrix(design_matrix, svd_rank, prop_var, &u, &svd_design_matrix_v);
	col_means_beta = approximate_beta_means(f_draws, u, svd_design_matrix_v);
	Eigen::MatrixXd Sigma_star = project_f_draws(f_draws, u);
	u.resize(0, 0);

	v_Sigma_star = svd_design_matrix_v * Sigma_star.triangularView<Eigen::Lower>();
	Lambda_chol = decompose_covariance_approximation(Sigma_star, svd_design_matrix_v, svd_rank);
	Sigma_star.resize(0, 0);

	Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(n_snps, n_snps);
	Lambda.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda_chol);
	Lambda_f = Lambda.triangularView<Eigen::Lower>() * v_Sigma_star;
	flat_Lambda = flatten_lambda(Lambda);

	svd_design_matrix_v.transposeInPlace();
    }
    f_draws.resize(0, 0);
    design_matrix.resize(0, 0);

    // Already known (argument)
    size_t flat_Lambda_size = n_snps * (n_snps + 1)/2;

    size_t Sigma_star_rows = v_Sigma_star.rows();
    size_t Sigma_star_cols = v_Sigma_star.cols();
    size_t Lambda_chol_rows = Lambda_chol.rows();
    size_t Lambda_chol_cols = Lambda_chol.cols();
    size_t Lambda_f_rows = Lambda_f.rows();
    size_t Lambda_f_cols = Lambda_f.cols();
    size_t svd_design_matrix_v_rows = svd_design_matrix_v.rows();
    size_t svd_design_matrix_v_cols = svd_design_matrix_v.cols();

    // Broadcast sizes
    MPI_Bcast(&Sigma_star_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Sigma_star_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_chol_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_chol_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_f_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_f_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&svd_design_matrix_v_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&svd_design_matrix_v_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Initialize ranges for MPI
    size_t n_snps_per_task = std::floor(n_snps/(double)n_tasks);
    size_t start_id = rank * n_snps_per_task;
    size_t end_id = std::min(n_snps, (rank + 1) * n_snps_per_task);
    if (rank == (n_tasks - 1)) {
	n_snps_per_task += n_snps - n_snps_per_task*n_tasks;
	end_id = std::min(n_snps, (rank + 1) * n_snps_per_task);
    }

    if (rank != 0) {
	v_Sigma_star.resize(Sigma_star_rows, Sigma_star_cols);
	Lambda_f.resize(Lambda_f_rows, Lambda_f_cols);
	Lambda_chol.resize(Lambda_chol_rows, Lambda_chol_cols);
	svd_design_matrix_v.resize(svd_design_matrix_v_rows, svd_design_matrix_v_cols);
	col_means_beta.resize(n_snps_per_task);
	flat_Lambda.resize(flat_Lambda_size);
    }

    // Eigen::VectorXd col_means_beta_partial(n_snps_per_task);

    {
	// Initializes the displacements and bufcounts.
	int displacements[1024];
	int bufcounts[1024] = { 0 };

	uint32_t sent_so_far = 0;
	uint32_t n_obs_per_task = std::floor(n_snps/n_tasks);
	for (uint16_t i = 0; i < n_tasks - 1; ++i) {
	    displacements[i] = sent_so_far;
	    bufcounts[i] = n_obs_per_task;
	    sent_so_far += bufcounts[i];
	}
	displacements[n_tasks - 1] = sent_so_far;
	bufcounts[n_tasks - 1] = n_snps - sent_so_far;
	bufcounts[0] = 0;

	MPI_Scatterv(col_means_beta.data(), bufcounts, displacements, MPI_DOUBLE, col_means_beta.data(), n_snps_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    {
	// Initializes the displacements and bufcounts.
	int displacements[1024];
	int bufcounts[1024] = { 0 };

	uint32_t sent_so_far = 0;
	for (uint16_t i = 0; i < n_tasks - 1; ++i) {
	    displacements[i] = sent_so_far;
	    bufcounts[i] = n_snps_per_task * svd_design_matrix_v_rows;
	    sent_so_far += bufcounts[i];
	}
	displacements[n_tasks - 1] = sent_so_far;
	bufcounts[n_tasks - 1] = (n_snps * svd_design_matrix_v_rows) - sent_so_far;
	bufcounts[0] = 0;

	MPI_Scatterv(svd_design_matrix_v.data(), bufcounts, displacements, MPI_DOUBLE, svd_design_matrix_v.data(), n_snps_per_task * svd_design_matrix_v_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Broadcast variables needed by all processes
    MPI_Bcast(v_Sigma_star.data(), v_Sigma_star.rows()*v_Sigma_star.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Lambda_f.data(), Lambda_f.rows()*Lambda_f.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_Lambda.data(), flat_Lambda_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Lambda_chol.data(), Lambda_chol.rows()*Lambda_chol.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> log_KLD_partial(n_snps_per_task);
    for (size_t i = 0; i < n_snps_per_task; ++i) {
	log_KLD_partial[i] = dropped_predictor_kld_lowrank(flat_Lambda, Lambda_f, Lambda_chol, v_Sigma_star, svd_design_matrix_v.col(i), col_means_beta[i], start_id + i);
    }

    std::vector<double> KLD_partial;
    std::transform(log_KLD_partial.begin(), log_KLD_partial.end(), std::back_inserter(KLD_partial), static_cast<double(*)(double)>(std::exp));
    double KLD_sum_local = std::accumulate(KLD_partial.begin(), KLD_partial.end(), 0.0);
    double KLD_sum_global = 0.0;
    MPI_Allreduce(&KLD_sum_local, &KLD_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> RATE_partial = rate_from_kld(log_KLD_partial, KLD_sum_global);

    std::vector<double> KLD(0);
    std::vector<double> RATE(0);
    if (rank == 0) {
	KLD.resize(n_snps);
	RATE.resize(n_snps);
    }

    {
	// Initializes the displacements and bufcounts.
	int displacements[1024];
	int bufcounts[1024] = { 0 };

	uint32_t sent_so_far = 0;
	uint32_t n_obs_per_task = std::floor(n_snps/n_tasks);
	for (uint16_t i = 0; i < n_tasks - 1; ++i) {
	    displacements[i] = sent_so_far;
	    bufcounts[i] = n_obs_per_task;
	    sent_so_far += bufcounts[i];
	}
	displacements[n_tasks - 1] = sent_so_far;
	bufcounts[n_tasks - 1] = n_snps - sent_so_far;

	MPI_Gatherv(&KLD_partial.front(), n_snps_per_task, MPI_DOUBLE, &KLD.front(), bufcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(&RATE_partial.front(), n_snps_per_task, MPI_DOUBLE, &RATE.front(), bufcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
	return RATEd(KLD, RATE);
    } else {
	return RATEd();
    }
}

#endif
