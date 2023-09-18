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

#include "bxzstr.hpp"

#include "CppRateRes.hpp"

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
    if (CmdOptionPresent(argv, argv+argc, "--beta-draws") && CmdOptionPresent(argv, argv+argc, "--fullrank")) {
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

void read_nonlinear(size_t *n_snps, std::istream *f_draws_file, std::istream *design_matrix_file, Eigen::MatrixXd *f_draws_mat, Eigen::SparseMatrix<double> *design_matrix_mat) {
    // TODO split into read_f_draws and read_design_matrix
    size_t n_obs = 0;
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
	++n_obs;
    }
    *design_matrix_mat = std::move(vec_to_sparse_matrix<double, bool>(X, n_obs, *n_snps));

    size_t n_f_draws = 0;
    std::vector<double> f_draws;
    while (std::getline(*f_draws_file, line)) {
	std::stringstream parts(line);
	std::string part;
	while(std::getline(parts, part, ',')) {
	    f_draws.emplace_back(std::stold(part));
	}
	++n_f_draws;
    }
    *f_draws_mat = std::move(vec_to_dense_matrix(f_draws, n_f_draws, n_obs));
}

void read_beta_draws(size_t *n_snps, std::istream *beta_draws_file, Eigen::MatrixXd *beta_draws_mat) {
    std::vector<double> beta_draws;
    size_t n_posterior_draws = 0;
    std::string line;
    bool first_line = true;
    while (std::getline(*beta_draws_file, line)) {
	if (first_line) {
	    *n_snps = std::count(line.begin(), line.end(), ',') + 1;
	    first_line = false;
	}
	std::stringstream parts(line);
	std::string part;
	while(std::getline(parts, part, ',')) {
	    beta_draws.emplace_back(std::stold(part));
	}
	++n_posterior_draws;
    }
    *beta_draws_mat = std::move(vec_to_dense_matrix(beta_draws, n_posterior_draws, *n_snps));
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

  bool from_beta_draws = CmdOptionPresent(argv, argv+argc, "--beta-draws");
  bool lowrank_beta_draws = CmdOptionPresent(argv, argv+argc, "--beta-draws") && CmdOptionPresent(argv, argv+argc, "-x");

  size_t n_snps = 0;

  Eigen::MatrixXd posterior_draws;
  Eigen::SparseMatrix<double> design_matrix;
  if (!from_beta_draws) {
      bxz::ifstream in(args.value<std::string>('f'));
      bxz::ifstream in2(args.value<std::string>('x'));
      read_nonlinear(&n_snps, &in, &in2, &posterior_draws, &design_matrix);
      in.close();
      in2.close();
      if (args.value<bool>("fullrank")) {
	  posterior_draws = std::move(nonlinear_coefficients(design_matrix, posterior_draws));
	  design_matrix.resize(0, 0);
      }
  } else if (from_beta_draws && !lowrank_beta_draws) {
      bxz::ifstream in(args.value<std::string>("beta-draws"));
      read_beta_draws(&n_snps, &in, &posterior_draws);
      in.close();
  } else if (from_beta_draws && lowrank_beta_draws) {
      bxz::ifstream in(args.value<std::string>("beta-draws"));
      bxz::ifstream design_matrix_file(args.value<std::string>('x'));
      read_beta_draws(&n_snps, &in, &posterior_draws);
      in.close();

      size_t n_obs = 0;
      std::vector<bool> X;
      std::string line;
      bool first_line = true;
      while (std::getline(design_matrix_file, line)) {
	  if (first_line) {
	      first_line = false;
	  }
	  std::stringstream parts(line);
	  std::string part;
	  while(std::getline(parts, part, ',')) {
	      X.emplace_back((bool)std::stol(part));
	  }
	  ++n_obs;
      }
      design_matrix = std::move(vec_to_sparse_matrix<double, bool>(X, n_obs, n_snps));
      design_matrix_file.close();
  }

  size_t id_start = std::max((args.value<size_t>("id-start") == 0 ? 0 : args.value<size_t>("id-start") - 1), (size_t)0);
  size_t id_end = std::min((args.value<size_t>("id-end") == 0 ? n_snps : args.value<size_t>("id-end")), n_snps);

  // TODO check that the snp ids in ids-to-test don't exceed the total n_snps or under 0

  RATEd res;
  if (args.value<bool>("fullrank") && !from_beta_draws) {
      res = RATE_beta_draws(posterior_draws, args.value<std::vector<size_t>>("ids-to-test"), id_start, id_end, n_snps, n_threads, n_threads_per_snp);
  } else if (!from_beta_draws) {
      size_t svd_rank = args.value<size_t>("low-rank") == 0 ? std::min(design_matrix.rows(), design_matrix.cols()) : args.value<size_t>("low-rank");
      res = RATE_lowrank(posterior_draws, design_matrix, args.value<std::vector<size_t>>("ids-to-test"), id_start, id_end, n_snps, svd_rank, args.value<double>("prop-var"), n_threads, n_threads_per_snp);
  } else if (from_beta_draws && lowrank_beta_draws) {
      // Calculate predictions = X*B' to use the lowrank approximation
      size_t svd_rank = args.value<size_t>("low-rank") == 0 ? std::min(design_matrix.rows(), design_matrix.cols()) : args.value<size_t>("low-rank");
      Eigen::MatrixXd predictions = posterior_draws * design_matrix.transpose();
      res = RATE_lowrank(predictions, design_matrix, args.value<std::vector<size_t>>("ids-to-test"), id_start, id_end, n_snps, svd_rank, args.value<double>("prop-var"), n_threads, n_threads_per_snp);
  } else if (from_beta_draws && args.value<bool>("fullrank")) {
      res = RATE_beta_draws(posterior_draws, args.value<std::vector<size_t>>("ids-to-test"), id_start, id_end, n_snps, n_threads, n_threads_per_snp);
  }

  std::cout << "#ESS: " << res.ESS << '\n';
  std::cout << "#Delta: " << res.Delta << '\n';
  std::cout << "#snp_id\tRATE\tKLD\n";
  for (size_t i = 0; i < n_snps; ++i) {
      std::cout << i << '\t' << res.RATE[i] << '\t' << res.KLD[i] << '\n';
  }
  std::cout << std::endl;

  return 0;
}
