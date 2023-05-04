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
#include "transformed_results_unit_test.hpp"

#include <cmath>

#include "CppRateRes.hpp"

TEST_F(TransformedResultsTest, RATE) {
    const std::vector<double> &RATE_got = rate_from_kld(this->KLD, this->KLD_sum);

    // Check dimensions of `RATE_got`
    EXPECT_EQ(RATE_got.size(), this->n_design_dim);

    // Check contents of `RATE_got`
    for (size_t i = 0; i < this->n_design_dim; ++i) {
	EXPECT_NEAR(RATE_got[i], this->RATE_expected[i], this->test_tolerance);
    }
}

TEST_F(TransformedResultsTest, Delta) {
    const std::vector<double> &RATE = rate_from_kld(this->KLD, this->KLD_sum);

    const double delta_got = rate_delta(RATE);
    EXPECT_NEAR(delta_got, this->Delta_expected, this->test_tolerance);
}

TEST_F(TransformedResultsTest, ESS) {
    const std::vector<double> &RATE = rate_from_kld(this->KLD, this->KLD_sum);
    const double delta = rate_delta(RATE);

    const double ess_got = delta_to_ess(delta);
    EXPECT_NEAR(ess_got, this->ESS_expected, this->test_tolerance);
}
