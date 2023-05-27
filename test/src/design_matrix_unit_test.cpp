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
#include "design_matrix_unit_test.hpp"

#include <cmath>

#include "util.hpp"
#include "lowrank.hpp"

// Test cpprate
TEST_F(DesignMatrixTest, vec_to_sparse_matrix) {
    const Eigen::SparseMatrix<double> &mat = vec_to_sparse_matrix<double, bool>(this->design_matrix, this->n_obs, this->n_design_dim);
    for (size_t i = 0; i < this->n_obs; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_EQ(this->design_matrix[i*n_design_dim + j], mat.coeff(i, j));
	}
    }
}

TEST_F(DecomposeDesignMatrixTest, svd_scaled_u) {
    Eigen::MatrixXd svd_u_got;
    Eigen::MatrixXd svd_v;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    decompose_design_matrix(this->sparse_design_matrix, svd_rank, this->prop_var, &svd_u_got, &svd_v);

    size_t num_nonzero_dims = svd_u_got.rows();

    // Check that the correct number of dimensions are dropped
    EXPECT_EQ(this->num_nonzero_dims_expected, num_nonzero_dims);

    // Check dimensions of `u`
    EXPECT_EQ(this->num_nonzero_dims_expected, svd_u_got.rows());
    EXPECT_EQ(this->n_obs, svd_u_got.cols());

    // Check contents of `u`
    for (size_t i = 0; i < this->num_nonzero_dims_expected; ++i) {
	for(size_t j = 0; j < this->n_obs; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_u_got(i, j)), std::abs(svd_u_expected[i*this->n_obs + j]), this->test_tolerance);
	}
    }
}

TEST_F(DecomposeDesignMatrixTest, svd_dropped_v) {
    Eigen::MatrixXd svd_u;
    Eigen::MatrixXd svd_v_got;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    decompose_design_matrix(this->sparse_design_matrix, svd_rank, this->prop_var, &svd_u, &svd_v_got);

    size_t num_nonzero_dims = svd_v_got.cols();

    // Check that the correct number of dimensions are dropped
    EXPECT_EQ(this->num_nonzero_dims_expected, num_nonzero_dims);

    // Check dimensions of `v`
    EXPECT_EQ(this->n_design_dim, svd_v_got.rows());
    EXPECT_EQ(this->num_nonzero_dims_expected, svd_v_got.cols());

    // Check contents of `v`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->num_nonzero_dims_expected; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_v_got(i, j)), std::abs(svd_v_expected[i*this->num_nonzero_dims_expected + j]), this->test_tolerance);
	}
    }
}
