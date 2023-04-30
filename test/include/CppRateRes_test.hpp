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
#ifndef CPPRATE_CPPRATERES_TEST_HPP
#define CPPRATE_CPPRATERES_TEST_HPP

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

// Integration test for cpprate
//
class CppRateResTest : public ::testing::Test {
  protected:
    void SetUp() override {
	// Parameters
	this->n_design_dim = 4;
	this->n_obs = 2;

	// Input data
	this->design_matrix = { 1,0,0,0,
				0,1,0,1 };
    }

    void TearDown() override {
	this->n_design_dim = 0;
	this->n_obs = 0;

	this->design_matrix.clear();
	this->design_matrix.shrink_to_fit();
    }

    // Test parameters
    size_t n_design_dim;
    size_t n_obs;

    // Input values
    std::vector<bool> design_matrix; // 10x`n_design_dims` matrix stored contiguously
    
    // Expected values
    static std::vector<double> svd_d;
    static std::vector<double> svd_u;
    static std::vector<double> svd_v;    
};
std::vector<double> CppRateResTest::svd_d = { 2.547924, 2.145761, 2.051606, 1.531563, 1.414214, 1.251116, 1.156189, 0.6685457 };
std::vector<double> CppRateResTest::svd_u = { 0,-0.02708703,-0.09722747,-0.01012977,-0.008285284,0,-0.8878288,-0.4367377,-0.1032576,-9.896359e-17,0,-0.488972,-0.1304165,-0.5545557,-0.4006958,0,0.2433472,-0.3396424,-0.3181729,-2.955358e-16,0,-0.2047639,0.304299,-0.393245,-0.2707039,0,-0.3753526,0.6722234,0.2116075,2.74614e-16,0,-0.3835255,0.4478546,0.4209023,0.1081235,0,-0.04236565,0.1548174,-0.6616054,-8.384766e-16,0,-2.394163e-14,-1.045776e-13,-3.227899e-15,-2.53389e-15,0,-8.920466e-13,-4.550291e-13,-1.094064e-13,1,0,-0.3090962,-0.6839341,0.4643097,-0.3570526,0,-0.06704195,0.2973115,0.02710943,-6.895913e-15,0,-0.1098872,-0.4407665,-0.3589628,0.7069234,0,-0.06268789,0.2923278,-0.2750806,-1.209058e-15,0,-0.6808355,0.1353726,0.1269928,0.3566169,0,0.0378603,-0.21024,0.5737597,5.52921e-17 };
std::vector<double> CppRateResTest::svd_v = { -0.03815949,-0.0178585,-0.3484518,-0.3484518,0,-0.3484518,-0.2119354,-0.003975696,-0.3484518,-0.05115721,-0.1714092,-1.680367e-16,-0.5198611,-0.003975696,-1.680367e-16,0,-0.003251779,-0.2095687,-0.3484518,0,-0.06077869,-0.6730589,0.1134084,0.1134084,0,0.1134084,-0.3065651,-0.2584425,0.1134084,-0.3761579,-0.1582853,-4.55299e-16,-0.04487692,-0.2584425,-4.55299e-16,0,-0.1867383,-0.219064,0.1134084,0,0.1483223,-0.4234306,-0.1829555,-0.1829555,0,-0.1829555,0.4307995,-0.1916766,-0.1829555,0.003335716,0.3276571,1.225736e-16,0.1447017,-0.1916766,1.225736e-16,0,-0.1319473,0.4759795,-0.1829555,0,0.2924166,0.09500116,-0.0276617,-0.0276617,0,-0.0276617,-0.3308959,0.2748187,-0.0276617,-0.6823948,0.1010845,-3.476018e-16,0.07342282,0.2748187,-3.476018e-16,0,0.07059679,0.3935012,-0.0276617,0,-7.391679e-14,-2.096033e-14,-6.307601e-13,-6.307601e-13,0,-6.307601e-13,-3.990289e-13,-2.299518e-15,-6.307601e-13,-9.42908e-14,-3.216791e-13,0.7071068,-9.524464e-13,-2.299518e-15,0.7071068,0,-1.738469e-15,-3.956788e-13,-6.307601e-13,0,-0.546659,-0.1613272,-0.05358571,-0.05358571,0,-0.05358571,0.2593051,0.3711163,-0.05358571,-0.2253881,0.2376369,-5.538831e-15,0.1840512,0.3711163,-5.538831e-15,0,-0.2853872,-0.3090221,-0.05358571,0,-0.3812234,0.2059121,-0.0542194,-0.0542194,0,-0.0542194,0.01491728,-0.3104706,-0.0542194,-0.3329626,0.2528373,-9.224642e-16,0.1986179,-0.3104706,-9.224642e-16,0,0.6114253,-0.1283862,-0.0542194,0,0.2024882,-0.2950072,0.05663084,0.05663084,0,0.05663084,0.5437469,0.1899539,0.05663084,-0.1601624,-0.3144737,-3.637005e-17,-0.2578428,0.1899539,-3.637005e-17,0,0.5334219,-0.1119855,0.05663084,0 };

#endif
