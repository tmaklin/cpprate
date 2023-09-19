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
#include "cxxargs.hpp"

#include <fstream>
#include <string>
#include <cstddef>
#include <algorithm>
#include <exception>

#include "bxzstr.hpp"
#include "BS_thread_pool.hpp"

#include "CppRateRes.hpp"
#include "CovarianceMatrix.hpp"
#include "RATE_res.hpp"

bool CmdOptionPresent(char **begin, char **end, const std::string &option) {
  return (std::find(begin, end, option) != end);
}

bool parse_args(int argc, char* argv[], cxxargs::Arguments &args) {
    args.add_short_argument<std::string>('f', "f-draws file (comma separated)");
    args.add_short_argument<std::string>('x', "design matrix (comma separated)");
    args.add_long_argument<std::string>("beta-draws", "beta-draws file (comma separated)");
    args.add_short_argument<size_t>('t', "Number of SNPs to process in parallel (default: 1)", 1);
    args.add_long_argument<size_t>("threads-per-snp", "Number of threads to use per SNP (default: 1)", 1);
    args.add_long_argument<std::vector<size_t>>("ids-to-test", "Comma-separated list of variable ids to test (optional)", std::vector<size_t>());
    args.add_long_argument<size_t>("id-start", "Test variables with id equal to or higher than `id-start` (optional)", 0);
    args.add_long_argument<size_t>("id-end", "Test variables with id equal to or less than `id-end` (optional)", 0);
    args.add_long_argument<double>("prop-var", "Proportion of variance to explain in lowrank factorization (default: 100%)", 1.1);
    args.add_long_argument<size_t>("low-rank", "Rank of the low-rank factorization (default: min(design_matrix.rows(), design_matrix.cols()))", 0);
    args.add_long_argument<bool>("fullrank", "Run fullrank algorithm (default: false)", false);
    args.add_long_argument<bool>("help", "Print the help message.", false);
    if (CmdOptionPresent(argv, argv+argc, "--help")) {
	std::cout << "\n" + args.help() << '\n' << std::endl;
	return true;
    }

    if (CmdOptionPresent(argv, argv+argc, "--ids-to-test") && (CmdOptionPresent(argv, argv+argc, "--id-start") || CmdOptionPresent(argv, argv+argc, "--id-end"))) {
	throw std::runtime_error("--ids-to-test and --id-start/--id-end are mutually exclusive");
    }

    // TODO stop if all three are present
    if (CmdOptionPresent(argv, argv+argc, "--beta-draws") && (!CmdOptionPresent(argv, argv+argc, "-x") || args.value<bool>("fullrank"))) {
	args.set_not_required('f');
	args.set_not_required('x');
    } else if (CmdOptionPresent(argv, argv+argc, "--beta-draws") && CmdOptionPresent(argv, argv+argc, "-x")) {
	args.set_not_required('f');
    } else {
	args.set_not_required("beta-draws");
    }

    args.parse(argc, argv);
    return false;
}

void read_design_matrix(std::istream *design_matrix_file, size_t *n_snps, size_t *n_obs, Eigen::SparseMatrix<double> *design_matrix_mat) {
    // Reset just in case
    *n_obs = 0;
    *n_snps = 0;

    std::vector<bool> X;
    std::string line;
    bool first_line = true;
    while (std::getline(*design_matrix_file, line)) {
	if (first_line) {
	    *n_snps = std::count(line.begin(), line.end(), ',') + 1;
	    first_line = false;
	}
	std::stringstream parts(line);
	std::string part;
	while(std::getline(parts, part, ',')) {
	    X.emplace_back((bool)std::stol(part));
	}
	++(*n_obs);
    }
    *design_matrix_mat = std::move(vec_to_sparse_matrix<double, bool>(X, *n_obs, *n_snps));
}

void read_posterior_draws(std::istream *posterior_draws_file, size_t *n_draws, size_t *n_obs, Eigen::MatrixXd *posterior_draws_mat) {
    *n_draws = 0;
    *n_obs = 0;
    std::vector<double> posterior_draws;
    std::string line;
    bool first_line = true;
    while (std::getline(*posterior_draws_file, line)) {
	if (first_line) {
	    *n_obs = std::count(line.begin(), line.end(), ',') + 1;
	    first_line = false;
	}
	std::stringstream parts(line);
	std::string part;
	while(std::getline(parts, part, ',')) {
	    posterior_draws.emplace_back(std::stold(part));
	}
	++(*n_draws);
    }
    *posterior_draws_mat = std::move(vec_to_dense_matrix(posterior_draws, *n_draws, *n_obs));
}

void read_nonlinear_coefficients(std::istream *design_matrix_file, size_t *n_snps, size_t *n_obs, Eigen::MatrixXd *posterior_draws) {
    Eigen::SparseMatrix<double> design_matrix;
    read_design_matrix(design_matrix_file, n_snps, n_obs, &design_matrix);
    *posterior_draws = std::move(nonlinear_coefficients(design_matrix, *posterior_draws));
}

void print_results(const RATEd &res, const size_t n_snps) {
    std::cout << "#ESS: " << res.ESS << '\n';
    std::cout << "#Delta: " << res.Delta << '\n';
    std::cout << "#snp_id\tRATE\tKLD\n";
    for (size_t i = 0; i < n_snps; ++i) {
	std::cout << i << '\t' << res.RATE[i] << '\t' << res.KLD[i] << '\n';
    }
    std::cout << std::endl;
}

Eigen::SparseMatrix<double> read_decomposition(const std::string &design_matrix_file, const size_t svd_rank, const double prop_var,
			size_t *n_snps, size_t *n_obs, Eigen::MatrixXd *svd_U, Eigen::MatrixXd *svd_V) {
      Eigen::SparseMatrix<double> design_matrix;
      bxz::ifstream design_matrix_in(design_matrix_file);
      read_design_matrix(&design_matrix_in, n_snps, n_obs, &design_matrix);
      design_matrix_in.close();

      size_t decomposition_rank = (svd_rank == 0 ? std::min(*n_snps, *n_obs) : svd_rank);
      decompose_design_matrix(design_matrix, decomposition_rank, prop_var, svd_U, svd_V);

      return design_matrix;
}

int main(int argc, char* argv[]) {
  cxxargs::Arguments args("cpprate", "Usage: cpprate -f <f-draws> -x <design-matrix> -n <num-obs> -d <num-SNPs> -m <num-f-draws>");
  try {
    bool quit = parse_args(argc, argv, args);
    if (quit) {
	return 1;
    }
  } catch (std::exception &e) {
    std::cerr << "Parsing arguments failed:\n"
	<< std::string("\t") + std::string(e.what()) + "\n"
	<< "\trun cpprate with the --help option for usage instructions.\n";
    std::cerr << std::endl;
    return 1;
  }

  size_t n_threads = args.value<size_t>('t');
  size_t n_threads_per_snp = args.value<size_t>("threads-per-snp");

#if defined(CPPRATE_OPENMP_SUPPORT) && (CPPRATE_OPENMP_SUPPORT) == 1
  // Use all available threads to do the linear algebra
  omp_set_num_threads(n_threads * n_threads_per_snp);
#else
  // Warn about ignored argument
  if (n_threads_per_snp > 1) {
      std::cerr << "WARNING: the `--n-threads-per-snp` argument is ignored (cpprate was compiled without OpenMP support)." << std::endl;
  }
#endif

  // Check what mode we're running in
  bool from_beta_draws = CmdOptionPresent(argv, argv+argc, "--beta-draws");
  bool run_fullrank = args.value<bool>("fullrank") || (from_beta_draws && !CmdOptionPresent(argv, argv+argc, "-x"));

  size_t low_rank_rank = args.value<size_t>("low-rank");

  // If read in posterior draws for beta and running fullrank algorithm
  // then there is no need to read in anything else so just run RATE.
  std::vector<double> col_means_beta;
  std::shared_ptr<CovMat> cov_beta_ptr;
  if (run_fullrank) {
      // Read in the posterior draws
      Eigen::MatrixXd posterior_draws;
      size_t n_draws = 0;
      size_t n_obs_draws = 0;
      bxz::ifstream posterior_draws_in((from_beta_draws ? args.value<std::string>("beta-draws") : args.value<std::string>('f')));
      read_posterior_draws(&posterior_draws_in, &n_draws, &n_obs_draws, &posterior_draws);
      posterior_draws_in.close();

      // If running fullrank algorithm on f draws we also need to read the design matrix
      if (!from_beta_draws) {
	  // Read in the design matrix
	  size_t n_snps = 0;
	  size_t n_obs_X = 0;
	  bxz::ifstream design_matrix_in(args.value<std::string>('x'));
	  // Next call will overwrite the f_draws currently stored in posterior_draws
	  // with beta draws = f_draws * pseudoinverse(design_matrix)
	  read_nonlinear_coefficients(&design_matrix_in, &n_snps, &n_obs_X, &posterior_draws);

	  // Check that the input dimensions are correct
	  if (n_obs_X != n_obs_draws) {
	      throw std::runtime_error("Number of rows in file " + args.value<std::string>('x') + " (" + std::to_string(n_obs_X) + ") does not match number of columns in file " + args.value<std::string>('f') + " (" + std::to_string(n_obs_draws) + ").");
	  }
      }

      cov_beta_ptr.reset(new FullrankCovMat());
      static_cast<FullrankCovMat*>(cov_beta_ptr.get())->fill(posterior_draws);
      col_means_beta = std::move(col_means2(posterior_draws));
  }

  // Otherwise running lowrank algorithm
  size_t n_snps = 0;
  if (!run_fullrank) {
      // Read in the posterior draws
      Eigen::MatrixXd posterior_draws;
      size_t n_draws = 0;
      size_t n_obs_draws = 0;
      bxz::ifstream posterior_draws_in((from_beta_draws ? args.value<std::string>("beta-draws") : args.value<std::string>('f')));
      read_posterior_draws(&posterior_draws_in, &n_draws, &n_obs_draws, &posterior_draws);
      posterior_draws_in.close();

      cov_beta_ptr.reset(new LowrankCovMat());
      // Decompose the design matrix
      Eigen::MatrixXd svd_design_matrix_U;
      size_t n_obs;

      if (from_beta_draws) {
	  const Eigen::SparseMatrix<double> &design_matrix = read_decomposition(args.value<std::string>('x'), low_rank_rank, args.value<double>("prop-var"), &n_snps, &n_obs, &svd_design_matrix_U,  static_cast<LowrankCovMat*>(cov_beta_ptr.get())->get_svd_V_p()).transpose();
	  posterior_draws *= design_matrix ;
      } else {
	  read_decomposition(args.value<std::string>('x'), low_rank_rank, args.value<double>("prop-var"), &n_snps, &n_obs, &svd_design_matrix_U,  static_cast<LowrankCovMat*>(cov_beta_ptr.get())->get_svd_V_p()) ;
      }

      // Check that the input dimensions are correct
      if (n_snps != n_obs_draws && from_beta_draws) {
	  throw std::runtime_error("Number of columns in file " + args.value<std::string>('x') + " (" + std::to_string(n_snps) + ") does not match number of columns in file " + args.value<std::string>("beta-draws") + " (" + std::to_string(n_obs_draws) + ").");
      } else if (n_obs != n_obs_draws && !from_beta_draws) {
	  throw std::runtime_error("Number of rows in file " + args.value<std::string>('x') + " (" + std::to_string(n_obs) + ") does not match number of columns in file " + args.value<std::string>('f') + " (" + std::to_string(n_obs_draws) + ").");
      }

      // If lowrank decomposition rank was supplied set it to minimum of the dimension of design_matrix
      low_rank_rank = low_rank_rank == 0 ? std::min(n_snps, n_obs) : low_rank_rank;

      static_cast<LowrankCovMat*>(cov_beta_ptr.get())->construct(project_f_draws(posterior_draws, svd_design_matrix_U), low_rank_rank);
      col_means_beta = std::move(approximate_beta_means(posterior_draws, svd_design_matrix_U,  static_cast<LowrankCovMat*>(cov_beta_ptr.get())->get_svd_V()));
  }

  if (!run_fullrank) {
       static_cast<LowrankCovMat*>(cov_beta_ptr.get())->fill();
       static_cast<LowrankCovMat*>(cov_beta_ptr.get())->logarithmize_lambda();
       static_cast<LowrankCovMat*>(cov_beta_ptr.get())->logarithmize_svd_V();
  }
    
  BS::thread_pool pool(n_threads);
  std::vector<std::future<std::vector<double>>> thread_futures;

  size_t id_start = std::max((args.value<size_t>("id-start") == 0 ? 0 : args.value<size_t>("id-start") - 1), (size_t)0);
  size_t id_end = std::min((args.value<size_t>("id-end") == 0 ? n_snps : args.value<size_t>("id-end")), n_snps);

  std::vector<size_t> snps_per_rank(n_threads, std::floor((id_end - id_start)/(double)n_threads));
  size_t n_snps_remainder = (id_end - id_start) - n_threads * snps_per_rank[0];
  for (size_t i = 0; i < n_snps_remainder; ++i) {
      snps_per_rank[i] += 1;
  }

  size_t n_processed = 0;
  for (size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      size_t thread_start_at = n_processed;
      size_t thread_stop_at = snps_per_rank[thread_id] + n_processed;
      n_processed += snps_per_rank[thread_id];
      size_t thread_n_snps = thread_stop_at - thread_start_at;
      thread_futures.emplace_back(pool.submit(run_RATE, col_means_beta, cov_beta_ptr,
					      args.value<std::vector<size_t>>("ids-to-test"), thread_start_at, thread_stop_at, thread_n_snps, n_threads_per_snp));
  }

  size_t global_index = 0;
  std::vector<double> log_KLD_global(n_snps, -36.84136);
  for (size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      const std::vector<double> &log_KLD_local = thread_futures[thread_id].get();
      for (size_t i = 0; i < log_KLD_local.size(); ++i) {
	  log_KLD_global[global_index] = log_KLD_local[i];
	  ++global_index;
      }
  }

  // Print results
  print_results(RATEd(log_KLD_global), n_snps);

  return 0;
}
