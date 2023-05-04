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
#include "kld_unit_test.hpp"

#include <cmath>

#include "CppRateRes.hpp"

TEST_F(KldTest, sherman_r) {
    const Eigen::MatrixXd &sherman_r_got = sherman_r(lambda_fullrank, cov_fullrank.col(1), cov_fullrank.col(1));

    // Check dimensions of `sherman_r_got`
    EXPECT_EQ(sherman_r_got.rows(), this->n_design_dim);
    EXPECT_EQ(sherman_r_got.cols(), this->n_design_dim);

    // Check contents of `sherman_r_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	for (size_t j = 0; j < this->n_design_dim; ++j) {
	    EXPECT_NEAR(sherman_r_got(i, j), this->sherman_r_expected[i*this->n_design_dim + j], this->test_tolerance);
	}
    }
}

TEST_F(KldTest, dropped_predictor_kld) {
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(dropped_predictor_kld(lambda_fullrank, cov_fullrank, col_means_fullrank, i), this->expected_KLD[i], 1e-4);
    }
}
