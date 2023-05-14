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

#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "CppRateRes.hpp"

#include "cpprate_mpi_config.hpp"
#include "MpiHandler.hpp"

inline RATEd RATE_lowrank_mpi(Eigen::MatrixXd &f_draws, Eigen::SparseMatrix<double> &design_matrix, const size_t n_snps, const size_t svd_rank, const double prop_var) {
    // ## WARNING: Do not compile with -ffast-math

    // Setup MPI
    MpiHandler handler;
    const int rank = handler.get_rank();
    const int n_tasks = handler.get_n_tasks();

    Eigen::VectorXd col_means_beta;
    Eigen::SparseMatrix<double> Sigma_star;
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd Lambda_chol;
    Eigen::MatrixXd svd_design_matrix_v;
    if (rank == 0) {
	Eigen::MatrixXd u;
	decompose_design_matrix(design_matrix, svd_rank, prop_var, &u, &svd_design_matrix_v);
	Lambda = std::move(Eigen::MatrixXd::Zero(n_snps, n_snps));
	col_means_beta = std::move(approximate_beta_means(f_draws, u, svd_design_matrix_v));
	Sigma_star = project_f_draws(f_draws, u);
	Lambda_chol = decompose_covariance_approximation(Sigma_star, svd_design_matrix_v, svd_rank);
	Lambda.template selfadjointView<Eigen::Lower>().rankUpdate(Lambda_chol);
    }
    f_draws.resize(0, 0);
    design_matrix.resize(0, 0);

    size_t Sigma_star_rows = Sigma_star.rows();
    size_t Sigma_star_cols = Sigma_star.rows();
    size_t Lambda_rows = Lambda.rows();
    size_t Lambda_cols = Lambda.cols();
    size_t Lambda_chol_rows = Lambda_chol.rows();
    size_t Lambda_chol_cols = Lambda_chol.cols();
    size_t svd_design_matrix_v_rows = svd_design_matrix_v.rows();
    size_t svd_design_matrix_v_cols = svd_design_matrix_v.rows();

    // Broadcast sizes
    MPI_Bcast(&Sigma_star_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Sigma_star_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_chol_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Lambda_chol_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
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
	Lambda.resize(Lambda_rows, Lambda_cols);
	Lambda_chol.resize(Lambda_chol_rows, Lambda_chol_cols);
	svd_design_matrix_v.resize(svd_design_matrix_v_rows, svd_design_matrix_v_cols);
    }

    handler.initialize_2(n_snps);
    const int* displacements_2 = handler.get_displacements();
    const int* bufcounts_2 = handler.get_bufcounts();
    Eigen::VectorXd col_means_beta_partial(n_snps_per_task);
    MPI_Scatterv(col_means_beta.data(), bufcounts_2, displacements_2, MPI_DOUBLE, col_means_beta_partial.data(), n_snps_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Eigen::MatrixXd tmp_mat(Sigma_star_rows, Sigma_star_cols);
    if (rank == 0) {
	tmp_mat = std::move(Sigma_star);
	col_means_beta.resize(0);
    }

    // Broadcast variables needed by all processes
    MPI_Bcast(tmp_mat.data(), tmp_mat.rows()*tmp_mat.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Lambda.data(), Lambda.rows()*Lambda.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Lambda_chol.data(), Lambda_chol.rows()*Lambda_chol.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(svd_design_matrix_v.data(), svd_design_matrix_v.rows()*svd_design_matrix_v.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Sigma_star = std::move(tmp_mat.sparseView());
    tmp_mat.resize(0, 0);

    std::vector<double> log_KLD_partial(n_snps_per_task);
    size_t index = 0;
    for (size_t i = start_id; i < end_id; ++i) {
	log_KLD_partial[index] = dropped_predictor_kld_lowrank(Lambda, Lambda_chol, Sigma_star, svd_design_matrix_v, col_means_beta_partial[index], i);
	++index;
    }

    std::vector<double> KLD_partial;
    std::transform(log_KLD_partial.begin(), log_KLD_partial.end(), std::back_inserter(KLD_partial), static_cast<double(*)(double)>(std::exp));
    double KLD_sum_local = std::accumulate(KLD_partial.begin(), KLD_partial.end(), 0.0);
    double KLD_sum_global = 0.0;
    MPI_Allreduce(&KLD_sum_local, &KLD_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> RATE_partial = rate_from_kld(log_KLD_partial, KLD_sum_global);

    std::vector<double> KLD;
    std::vector<double> RATE;
    if (rank == 0) {
	KLD.resize(n_snps);
	RATE.resize(n_snps);
    }

    MPI_Gatherv(&KLD_partial.front(), n_snps_per_task, MPI_DOUBLE, &KLD.front(), bufcounts_2, displacements_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&RATE_partial.front(), n_snps_per_task, MPI_DOUBLE, &RATE.front(), bufcounts_2, displacements_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
	return RATEd(KLD, RATE);
    } else {
	return RATEd();
    }
}

#endif
