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
#ifndef CPPRATE_SVD_UNIT_TEST_HPP
#define CPPRATE_SVD_UNIT_TEST_HPP

#include <cstddef>
#include <vector>

#include <Eigen/SparseCore>

#include "gtest/gtest.h"

class SvdTest : public ::testing::Test {
protected:
    double test_tolerance = 1e-6; // Singular values don't pass 1e-7

    void SetUp() override {
	this->n_obs = 10;
	this->n_design_dim = 20;

	// Input design matrix
	std::vector<bool> in_vals = {
	    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	    0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
	    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
	    0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,
	    0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
	    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	    0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,
	    0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,
	    0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,
	    0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0
	};

	this->sparse_design_matrix.resize(this->n_obs, this->n_design_dim);
	for (size_t i = 0; i < this->n_obs; ++i) {
	    for (size_t j = 0; j < this->n_design_dim; ++j) {
		if (in_vals[i*this->n_design_dim + j]) {
		    this->sparse_design_matrix.insert(i, j) = (double)1;
		}
	    }
	}
    }

    void TearDown() override {
	this->n_obs = 0;
	this->n_design_dim = 0;
	this->sparse_design_matrix.resize(0, 0);
    }

    // Test parameters
    size_t n_design_dim;
    size_t n_obs;

    // Input values
    Eigen::SparseMatrix<double> sparse_design_matrix;

    // Expected values
    static std::vector<double> svd_u_expected;
    static std::vector<double> svd_v_expected;
    static std::vector<double> svd_d_expected;

};
std::vector<double> SvdTest::svd_u_expected = { 0,0,0,0,0,0,0,0,0,0,
					      -0.02708703,-0.488972,-0.2047639,-0.3835255,-2.394163e-14,-0.3090962,-0.1098872,-0.6808355,0,0,
					      -0.09722747,-0.1304165,0.304299,0.4478546,-1.045776e-13,-0.6839341,-0.4407665,0.1353726,0,0,
					      -0.01012977,-0.5545557,-0.393245,0.4209023,-3.227899e-15,0.4643097,-0.3589628,0.1269928,0,0,
					      -0.008285284,-0.4006958,-0.2707039,0.1081235,-2.53389e-15,-0.3570526,0.7069234,0.3566169,0,0,
					      0,0,0,0,0,0,0,0,0,0,
					      -0.8878288,0.2433472,-0.3753526,-0.04236565,-8.920466e-13,-0.06704195,-0.06268789,0.0378603,0,0,
					      -0.4367377,-0.3396424,0.6722234,0.1548174,-4.550291e-13,0.2973115,0.2923278,-0.21024,0,0,
					      -0.1032576,-0.3181729,0.2116075,-0.6616054,-1.094064e-13,0.02710943,-0.2750806,0.5737597,0,0,
					      -9.896359e-17,-2.955358e-16,2.74614e-16,-8.384766e-16,1,-6.895913e-15,-1.209058e-15,5.52921e-17,0,0 };
std::vector<double> SvdTest::svd_v_expected = { -0.03815949,-0.06077869,0.1483223,0.2924166,-7.391679e-14,-0.546659,-0.3812234,0.2024882,0,0,-0.0178585,-0.6730589,-0.4234306,0.09500116,-2.096033e-14,-0.1613272,0.2059121,-0.2950072,0,0,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0.05663084,0,0,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0.05663084,0,0,0,0,0,0,0,0,0,0,0,0,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0.05663084,0,0,-0.2119354,-0.3065651,0.4307995,-0.3308959,-3.990289e-13,0.2593051,0.01491728,0.5437469,0,0,-0.003975696,-0.2584425,-0.1916766,0.2748187,-2.299518e-15,0.3711163,-0.3104706,0.1899539,0,0,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0.05663084,0,0,-0.05115721,-0.3761579,0.003335716,-0.6823948,-9.42908e-14,-0.2253881,-0.3329626,-0.1601624,0,0,-0.1714092,-0.1582853,0.3276571,0.1010845,-3.216791e-13,0.2376369,0.2528373,-0.3144737,0,0,-1.680367e-16,-4.55299e-16,1.225736e-16,-3.476018e-16,0.7071068,-5.538831e-15,-9.224642e-16,-3.637005e-17,0,0,-0.5198611,-0.04487692,0.1447017,0.07342282,-9.524464e-13,0.1840512,0.1986179,-0.2578428,0,0,-0.003975696,-0.2584425,-0.1916766,0.2748187,-2.299518e-15,0.3711163,-0.3104706,0.1899539,0,0,-1.680367e-16,-4.55299e-16,1.225736e-16,-3.476018e-16,0.7071068,-5.538831e-15,-9.224642e-16,-3.637005e-17,0,0,0,0,0,0,0,0,0,0,0,0,-0.003251779,-0.1867383,-0.1319473,0.07059679,-1.738469e-15,-0.2853872,0.6114253,0.5334219,0,0,-0.2095687,-0.219064,0.4759795,0.3935012,-3.956788e-13,-0.3090221,-0.1283862,-0.1119855,0,0,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0.05663084,0,0,0,0,0,0,0,0,0,0,0,0 };
std::vector<double> SvdTest::svd_d_expected = { 2.547924, 2.145761, 2.051606, 1.531563, 1.414214, 1.251116, 1.156189, 0.6685457, 0, 0 };

#endif
