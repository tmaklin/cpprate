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

#include "CppRateRes.hpp"
#include "CppRateRes_mpi.hpp"

#include "cpprate_mpi_config.hpp"

bool CmdOptionPresent(char **begin, char **end, const std::string &option) {
  return (std::find(begin, end, option) != end);
}

void parse_args(int argc, char* argv[], cxxargs::Arguments &args) {
    args.add_short_argument<std::string>('f', "f-draws file (comma separated)");
    args.add_short_argument<std::string>('x', "design matrix (comma separated)");
    args.add_short_argument<size_t>('n', "Number of observations (rows in design matrix; columns in f-draws");
    args.add_short_argument<size_t>('d', "Number of SNPs tested (cols in design matrix");
    args.add_short_argument<size_t>('m', "Number of posterior samples (rows in f-draws)");
    args.add_short_argument<size_t>('t', "Number of threads to use (default: 1)", 1);
    args.add_long_argument<bool>("fullrank", "Run fullrank algorithm (default: false)", false);
    args.add_long_argument<size_t>("lowrank-dim", "Dimension of the lowrank approximation (default: min(n, d))", 0);
    args.add_long_argument<bool>("help", "Print the help message.", false);
    if (CmdOptionPresent(argv, argv+argc, "--help")) {
	std::cout << "\n" + args.help() << '\n' << std::endl;
  }
  args.parse(argc, argv);
}

int main(int argc, char* argv[]) {
  cxxargs::Arguments args("cpprate", "Usage: cpprate -f <f-draws> -x <design-matrix> -n <num-obs> -d <num-SNPs> -m <num-f-draws>");

  int rc = MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t num_threads = 0;
  if (rank == 0) {
  try {
    parse_args(argc, argv, args);
  } catch (std::exception &e) {
    std::cerr << "Parsing arguments failed:\n"
	<< std::string("\t") + std::string(e.what()) + "\n"
	<< "\trun cpprate with the --help option for usage instructions.\n";
    std::cerr << std::endl;
    return 1;
  }
  num_threads = args.value<size_t>('t');
  }
  MPI_Bcast(&num_threads, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  omp_set_num_threads(num_threads);

  size_t n_snps;
  size_t n_obs;

  Eigen::MatrixXd f_draws_mat;
  Eigen::SparseMatrix<double> design_matrix;
  if (rank == 0) {
  n_obs = args.value<size_t>('n');
  n_snps = args.value<size_t>('d');
  size_t n_f_draws = args.value<size_t>('m');

  std::vector<double> f_draws;
  std::ifstream in(args.value<std::string>('f'));
  std::string line;
  while (std::getline(in, line)) {
      std::stringstream parts(line);
      std::string part;
      while(std::getline(parts, part, ',')) {
	  f_draws.emplace_back(std::stold(part));
      }
  }
  in.close();

  f_draws_mat = std::move(vec_to_dense_matrix(f_draws, n_f_draws, n_obs));

  std::vector<bool> X;
  std::ifstream in2(args.value<std::string>('x'));
  while (std::getline(in2, line)) {
      std::stringstream parts(line);
      std::string part;
      while(std::getline(parts, part, ',')) {
	  X.emplace_back((bool)std::stol(part));
      }
  }
  in2.close();

  design_matrix = std::move(vec_to_sparse_matrix<double, bool>(X, n_obs, n_snps));
  }

  MPI_Bcast(&n_snps, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_obs, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  RATEd res;
  int fullrank = false;
  if (rank == 0) {
  fullrank = args.value<bool>("fullrank");
  }
  MPI_Bcast(&fullrank, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD); // Wait for all ranks
  if (fullrank) {
      res = RATE_fullrank(f_draws_mat, design_matrix, n_snps);
  } else {
      const double prop_var = 1.0; // (if prop.var == 1.0 the last component is always ignored)?
      size_t svd_rank = std::min(n_obs, n_snps);
      res = RATE_lowrank_mpi(f_draws_mat, design_matrix, n_snps, svd_rank, prop_var);
  }

  if (rank == 0) {
      std::cout << "#ESS: " << res.ESS << '\n';
      std::cout << "#Delta: " << res.Delta << '\n';
      std::cout << "#snp_id\tRATE\tKLD\n";
      for (size_t i = 0; i < n_snps; ++i) {
	  std::cout << i << '\t' << res.RATE[i] << '\t' << res.KLD[i] << '\n';
      }
      std::cout << std::endl;
  }

  rc = MPI_Finalize();

  return 0;
}