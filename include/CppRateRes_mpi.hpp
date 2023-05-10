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
    Eigen::MatrixXd svd_cov_beta_u;
    Eigen::MatrixXd svd_design_matrix_v;
    if (rank == 0) {
	Eigen::MatrixXd u;
	decompose_design_matrix(design_matrix, svd_rank, prop_var, &u, &svd_design_matrix_v);

	col_means_beta = std::move(approximate_beta_means(f_draws, u, svd_design_matrix_v));
	Sigma_star = project_f_draws(f_draws, u);
	svd_cov_beta_u = std::move(decompose_covariance_approximation(Sigma_star, svd_design_matrix_v, svd_rank).adjoint());
    }
    f_draws.resize(0, 0);
    design_matrix.resize(0, 0);

    size_t Sigma_star_rows = Sigma_star.rows();
    size_t Sigma_star_cols = Sigma_star.rows();
    size_t svd_cov_beta_u_rows = svd_cov_beta_u.rows();
    size_t svd_cov_beta_u_cols = svd_cov_beta_u.cols();
    size_t svd_design_matrix_v_rows = svd_design_matrix_v.rows();
    size_t svd_design_matrix_v_cols = svd_design_matrix_v.rows();

    // Broadcast sizes
    MPI_Bcast(&Sigma_star_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Sigma_star_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&svd_cov_beta_u_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&svd_cov_beta_u_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&svd_design_matrix_v_rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&svd_design_matrix_v_cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
	col_means_beta.resize(n_snps);
	Sigma_star.resize(Sigma_star_rows, Sigma_star_cols);
	svd_cov_beta_u.resize(svd_cov_beta_u_rows, svd_cov_beta_u_cols);
	svd_design_matrix_v.resize(svd_design_matrix_v_rows, svd_design_matrix_v_cols);
    }

    // Broadcast variables needed by all processes
    Eigen::MatrixXd tmp_mat = Sigma_star;
    MPI_Bcast(col_means_beta.data(), col_means_beta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmp_mat.data(), tmp_mat.rows()*tmp_mat.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(svd_cov_beta_u.data(), svd_cov_beta_u.rows()*svd_cov_beta_u.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(svd_design_matrix_v.data(), svd_design_matrix_v.rows()*svd_design_matrix_v.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Sigma_star = tmp_mat.sparseView();
    tmp_mat.resize(0, 0);

    // Initialize ranges for MPI
    size_t n_snps_per_task = std::floor(n_snps/(double)n_tasks);
    size_t start_id = rank * n_snps_per_task;
    size_t end_id = std::min(n_snps, (rank + 1) * n_snps_per_task);
    if (rank == (n_tasks - 1)) {
	n_snps_per_task += n_snps - n_snps_per_task*n_tasks;
	end_id = std::min(n_snps, (rank + 1) * n_snps_per_task);
    }

    std::vector<double> KLD_partial(n_snps_per_task);
    size_t index = 0;
    for (size_t i = start_id; i < end_id; ++i) {
	KLD_partial[index] = dropped_predictor_kld_lowrank(svd_cov_beta_u, Sigma_star, svd_design_matrix_v, col_means_beta[i], i);
	++index;
    }

    std::vector<double> KLD;
    if (rank == 0) {
	KLD.resize(n_snps);
    }

    handler.initialize_2(n_snps);
    const int* displacements_2 = handler.get_displacements();
    const int* bufcounts_2 = handler.get_bufcounts();
    MPI_Gatherv(&KLD_partial.front(), n_snps_per_task, MPI_DOUBLE, &KLD.front(), bufcounts_2, displacements_2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
	return RATEd(KLD);
    } else {
	return RATEd();
    }
}

#endif
