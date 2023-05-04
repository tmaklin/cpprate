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
#include "lowrank_integration_test.hpp"

#include "CppRateRes.hpp"

// Test lowrank ESS
TEST_F(LowrankIntegrationTest, LowRank_EssIsCorrect) {
    RATEd res = RATE(this->n_obs, this->n_design_dim, this->n_f_draws, this->design_matrix, this->f_draws, true, this->rank_r);
    EXPECT_NEAR(res.ESS, this->expected_lr_ESS, 1e-5);
}

// Test lowrank Delta
TEST_F(LowrankIntegrationTest, LowRank_DeltaIsCorrect) {
    RATEd res = RATE(this->n_obs, this->n_design_dim, this->n_f_draws, this->design_matrix, this->f_draws, true, this->rank_r);
    EXPECT_NEAR(res.Delta, this->expected_lr_Delta, 1e-6);
}

// Test lowrank RATE
TEST_F(LowrankIntegrationTest, LowRank_RateIsCorrect) {
    RATEd res = RATE(this->n_obs, this->n_design_dim, this->n_f_draws, this->design_matrix, this->f_draws, true, this->rank_r);
    for (size_t i = 0; i < n_design_dim; ++i) {
	EXPECT_NEAR(res.RATE[i], this->expected_lr_RATE[i], 1e-6);
    }

}

// Test lowrank KLD
TEST_F(LowrankIntegrationTest, LowRank_KldIsCorrect) {
    RATEd res = RATE(this->n_obs, this->n_design_dim, this->n_f_draws, this->design_matrix, this->f_draws, true, this->rank_r);
    for (size_t i = 0; i < n_design_dim; ++i) {
	EXPECT_NEAR(res.KLD[i], this->expected_lr_KLD[i], 1e-5);
    }
}
