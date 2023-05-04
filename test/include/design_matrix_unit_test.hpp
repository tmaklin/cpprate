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
#ifndef CPPRATE_DESIGN_MATRIX_UNIT_TEST_HPP
#define CPPRATE_DESIGN_MATRIX_UNIT_TEST_HPP

#include <cstddef>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "gtest/gtest.h"

class DesignMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
	this->n_obs = 10;
	this->n_design_dim = 20;

	// Input design matrix
	this->design_matrix = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
				0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
				1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
				0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,
				0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
				0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
				0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,
				0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,
				0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,
				0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0 };
    }

    void TearDown() override {
	this->n_obs = 0;
	this->n_design_dim = 0;
	this->design_matrix.clear();
	this->design_matrix.shrink_to_fit();
    }

    // Test parameters
    size_t n_design_dim;
    size_t n_obs;

    // Input values
    std::vector<bool> design_matrix; // 10x`n_design_dims` matrix stored contiguously

};

class DecomposeDesignMatrixTest : public ::testing::Test {
protected:
    double test_tolerance = 1e-7;

    void SetUp() override {
	this->n_obs = 10;
	this->n_design_dim = 20;
	this->prop_var = 1.0;

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
	this->prop_var = 0.0;
	this->sparse_design_matrix.resize(0, 0);
    }

    // Test parameters
    size_t n_design_dim;
    size_t n_obs;
    double prop_var;

    // Input values
    Eigen::SparseMatrix<double> sparse_design_matrix;

    // Expected values
    static size_t num_nonzero_dims_expected;
    static std::vector<double> svd_u_expected;
    static std::vector<double> svd_v_expected;

};
std::vector<double> DecomposeDesignMatrixTest::svd_u_expected = { 0,-0.01063102,-0.03815949,-0.003975696,-0.003251779,0,-0.3484518,-0.1714092,-0.04052619,-3.884087e-17,0,-0.2278781,-0.06077869,-0.2584425,-0.1867383,0,0.1134084,-0.1582853,-0.1482798,-1.377301e-16,0,-0.09980662,0.1483223,-0.1916766,-0.1319473,0,-0.1829555,0.3276571,0.1031423,1.338532e-16,0,-0.2504144,0.2924166,0.2748187,0.07059679,0,-0.0276617,0.1010845,-0.4319804,-5.474645e-16,0,-1.692929e-14,-7.394756e-14,-2.282469e-15,-1.791731e-15,0,-6.307722e-13,-3.217542e-13,-7.736201e-14,0.7071068,0,-0.2470563,-0.546659,0.3711163,-0.2853872,0,-0.05358571,0.2376369,0.02166819,-5.511808e-15,0,-0.09504256,-0.3812234,-0.3104706,0.6114253,0,-0.0542194,0.2528373,-0.23792,-1.045726e-15 };
std::vector<double> DecomposeDesignMatrixTest::svd_v_expected = { -0.03815949,-0.06077869,0.1483223,0.2924166,-7.391679e-14,-0.546659,-0.3812234,-0.0178585,-0.6730589,-0.4234306,0.09500116,-2.096033e-14,-0.1613272,0.2059121,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0,0,0,0,0,0,0,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,-0.2119354,-0.3065651,0.4307995,-0.3308959,-3.990289e-13,0.2593051,0.01491728,-0.003975696,-0.2584425,-0.1916766,0.2748187,-2.299518e-15,0.3711163,-0.3104706,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,-0.05115721,-0.3761579,0.003335716,-0.6823948,-9.42908e-14,-0.2253881,-0.3329626,-0.1714092,-0.1582853,0.3276571,0.1010845,-3.216791e-13,0.2376369,0.2528373,-1.680367e-16,-4.55299e-16,1.225736e-16,-3.476018e-16,0.7071068,-5.538831e-15,-9.224642e-16,-0.5198611,-0.04487692,0.1447017,0.07342282,-9.524464e-13,0.1840512,0.1986179,-0.003975696,-0.2584425,-0.1916766,0.2748187,-2.299518e-15,0.3711163,-0.3104706,-1.680367e-16,-4.55299e-16,1.225736e-16,-3.476018e-16,0.7071068,-5.538831e-15,-9.224642e-16,0,0,0,0,0,0,0,-0.003251779,-0.1867383,-0.1319473,0.07059679,-1.738469e-15,-0.2853872,0.6114253,-0.2095687,-0.219064,0.4759795,0.3935012,-3.956788e-13,-0.3090221,-0.1283862,-0.3484518,0.1134084,-0.1829555,-0.0276617,-6.307601e-13,-0.05358571,-0.0542194,0,0,0,0,0,0,0 };
size_t DecomposeDesignMatrixTest::num_nonzero_dims_expected = 7;

#endif
