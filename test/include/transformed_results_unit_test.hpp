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
#ifndef CPPRATE_TRANSFORMED_RESULTS_UNIT_TEST_HPP
#define CPPRATE_TRANSFORMED_RESULTS_UNIT_TEST_HPP

#include <cstddef>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "gtest/gtest.h"

class TransformedResultsTest : public ::testing::Test {
protected:
    double test_tolerance = 1e-7;

    void SetUp() override {
	this->n_design_dim = 20;

	this->KLD = { 46.7013731559339, 18.0326943349702, 11.3476008015389, 11.3476008015389, 0.0, 11.3476008015389, 6.92892677441328, 68.0601282048986, 11.3476008015389, 11.1261904144762, 153.373975215714, 0.439287705790222, 296.568244722871, 68.0601282048987, 0.439287705790222, 0, 205.01033657952, 1.93746322665465, 11.3476008015389, 0.0 };

	this->KLD_sum = 0.0;
	for (size_t i = 0; i < this->n_design_dim; ++i) {
	    KLD_sum += KLD[i];
	}

    }
    void TearDown() override {
	this->n_design_dim = 0;
	this->KLD_sum = 0.0;
	this->KLD.clear();
	this->KLD.shrink_to_fit();
    }

    // Test parameters
    size_t n_design_dim;

    // Input values
    std::vector<double> KLD;
    double KLD_sum;

    // Expected values
    static std::vector<double> RATE_expected;
    static double ESS_expected;
    static double Delta_expected;
};
std::vector<double> TransformedResultsTest::RATE_expected = { 0.0500327518940475, 0.0193190319828556, 0.0121570664228735, 0.0121570664228735, 0.0, 0.0121570664228735, 0.00742319231254111, 0.0729151045940945, 0.0121570664228735, 0.011919862027926, 0.164314698485404, 0.000470623694950496, 0.317723535843982, 0.0729151045940946, 0.000470623694950496, 0.0, 0.219634469238193, 0.00207566952259381, 0.0121570664228735, 0.0 };
double TransformedResultsTest::ESS_expected = 49.5899195201055;
double TransformedResultsTest::Delta_expected = 1.01653886450565;

#endif
