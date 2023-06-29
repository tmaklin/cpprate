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

#include "CppRateRes.hpp"

bool CmdOptionPresent(char **begin, char **end, const std::string &option) {
  return (std::find(begin, end, option) != end);
}

void parse_args(int argc, char* argv[], cxxargs::Arguments &args) {
    args.add_short_argument<std::string>('f', "f-draws file (comma separated)");
    args.add_short_argument<std::string>('x', "design matrix (comma separated)");
    args.add_long_argument<std::string>("beta-draws", "beta-draws file (comma separated)");
    args.add_short_argument<size_t>('t', "Number of threads to use (default: 1)", 1);
    args.add_long_argument<double>("prop-var", "Proportion of variance to explain in lowrank factorization (default: 100%)", 1.1);
    args.add_long_argument<size_t>("low-rank", "Rank of the low-rank factorization (default: min(design_matrix.rows(), design_matrix.cols()))", 0);
    args.add_long_argument<bool>("fullrank", "Run fullrank algorithm (default: false)", false);
    args.add_long_argument<bool>("help", "Print the help message.", false);
    if (CmdOptionPresent(argv, argv+argc, "--help")) {
	std::cout << "\n" + args.help() << '\n' << std::endl;
    }

    if (CmdOptionPresent(argv, argv+argc, "--beta-draws") && !CmdOptionPresent(argv, argv+argc, "--help")) {
	args.set_not_required('f');
	args.set_not_required('x');
    } else if (!CmdOptionPresent(argv, argv+argc, "--help")) {
	args.set_not_required("beta-draws");
    }

    args.parse(argc, argv);
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
    parse_args(argc, argv, args);
  } catch (std::exception &e) {
    std::cerr << "Parsing arguments failed:\n"
	<< std::string("\t") + std::string(e.what()) + "\n"
	<< "\trun cpprate with the --help option for usage instructions.\n";
    std::cerr << std::endl;
    return 1;
  }

  omp_set_num_threads(args.value<size_t>('t'));

  bool from_beta_draws = CmdOptionPresent(argv, argv+argc, "--beta-draws");

  size_t n_snps = 0;

  Eigen::MatrixXd posterior_draws;
  Eigen::SparseMatrix<double> design_matrix;
  if (!from_beta_draws) {
      std::ifstream in(args.value<std::string>('f'));
      std::ifstream in2(args.value<std::string>('x'));
      read_nonlinear(&n_snps, &in, &in2, &posterior_draws, &design_matrix);
      in.close();
      in2.close();
  } else {
      std::ifstream in(args.value<std::string>("beta-draws"));
      read_beta_draws(&n_snps, &in, &posterior_draws);
      in.close();
  }

  RATEd res;
  if (args.value<bool>("fullrank") && !from_beta_draws) {
      res = RATE_fullrank(posterior_draws, design_matrix, n_snps);
  } else if (!from_beta_draws) {
      size_t svd_rank = args.value<size_t>("low-rank") == 0 ? std::min(design_matrix.rows(), design_matrix.cols()) : args.value<size_t>("low-rank");
      res = RATE_lowrank(posterior_draws, design_matrix, n_snps, svd_rank, args.value<double>("prop-var"));
  } else if (from_beta_draws) {
      res = RATE_beta_draws(posterior_draws, n_snps);
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
