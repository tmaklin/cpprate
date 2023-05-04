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
#include "svd_unit_test.hpp"

#include <cmath>

#include <Eigen/Dense>

#include "CppRateRes.hpp"

TEST_F(SvdTest, svd_factor_u) {
    Eigen::MatrixXd svd_u_got;
    Eigen::MatrixXd svd_v;
    Eigen::VectorXd svd_d;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    svd(this->sparse_design_matrix, svd_rank, &svd_u_got, &svd_v, &svd_d);

    // Check dimensions of `u`
    EXPECT_EQ(svd_rank, svd_u_got.rows());
    EXPECT_EQ(svd_rank, svd_u_got.cols());

    // Check contents of `u`
    for (size_t i = 0; i < svd_rank; ++i) {
	for (size_t j = 0; j < svd_rank; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_u_got(i, j)), std::abs(svd_u_expected[i*svd_rank + j]), this->test_tolerance);
	}
    }
}

TEST_F(SvdTest, svd_factor_v) {
    Eigen::MatrixXd svd_u;
    Eigen::MatrixXd svd_v_got;
    Eigen::VectorXd svd_d;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    svd(this->sparse_design_matrix, svd_rank, &svd_u, &svd_v_got, &svd_d);

    // Check dimensions of `v`
    EXPECT_EQ(this->n_design_dim, svd_v_got.rows());
    EXPECT_EQ(svd_rank, svd_v_got.cols());

    // Check contents of `v`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < svd_rank; ++j) {
	    // The signs might be flipped.
	    EXPECT_NEAR(std::abs(svd_v_got(i, j)), std::abs(svd_v_expected[i*svd_rank + j]), this->test_tolerance);
	}
    }
}

TEST_F(SvdTest, svd_singular_values) {
    Eigen::MatrixXd svd_u;
    Eigen::MatrixXd svd_v;
    Eigen::VectorXd svd_d_got;

    size_t svd_rank = std::min(this->n_obs, this->n_design_dim);
    svd(this->sparse_design_matrix, svd_rank, &svd_u, &svd_v, &svd_d_got);

    // Check dimensions of `d`
    EXPECT_EQ(svd_rank, svd_d_got.size());

    // Check contents of `d`
    for (size_t i = 0; i < svd_rank; ++i) {
	EXPECT_NEAR(svd_d_got(i), svd_d_expected[i], this->test_tolerance);
    }
}
