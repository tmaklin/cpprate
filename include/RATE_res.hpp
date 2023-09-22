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
#ifndef CPPRATE_RATE_RES_HPP
#define CPPRATE_RATE_RES_HPP

#include <vector>
#include <cstddef>
#include <cmath>
#include <numeric>

#include "cpprate_openmp_config.hpp"

inline std::vector<double> rate_from_kld(const std::vector<double> &log_kld, const double kld_sum) {
    std::vector<double> RATE(log_kld.size());
    double log_kld_sum = std::log(kld_sum);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < log_kld.size(); ++i) {
	RATE[i] = std::exp(log_kld[i] - log_kld_sum);
    }
    return RATE;
}

inline std::vector<double> rate_from_log_kld(const std::vector<double> &log_kld) {
    double max_elem = 0.0;
    // TODO pragma with custom reduction to find maximum
    for (size_t i = 0; i < log_kld.size(); ++i) {
	max_elem = (max_elem > log_kld[i] ? max_elem : log_kld[i]);
    }
    double tmp_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:tmp_sum)
    for (size_t i = 0; i < log_kld.size(); ++i) {
	tmp_sum += std::exp(log_kld[i] - max_elem);
    }
    double log_kld_sum = std::log(tmp_sum) + max_elem;

    std::vector<double> RATE(log_kld.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < log_kld.size(); ++i) {
	RATE[i] = std::exp(log_kld[i] - log_kld_sum);
    }
    return RATE;
}

inline double rate_delta(const std::vector<double> &RATE) {
    double Delta = 0.0;
    size_t num_snps = RATE.size();
#pragma omp parallel for schedule(static) reduction(+:Delta)
    for (size_t i = 0; i < num_snps; ++i) {
	Delta += RATE[i]*(std::log(num_snps) + std::log(RATE[i] + 1e-16));
    }

    return Delta;
}

inline double delta_to_ess(const double delta) {
    return std::exp(std::log(1.0) - std::log(1.0 + delta))*100.0;
}

struct RATEd {
public:
    double ESS;
    double Delta;
    std::vector<double> RATE;
    std::vector<double> KLD;

    RATEd() = default;

    RATEd(double _ESS, double _Delta, std::vector<double> _RATE, std::vector<double> _KLD) {
	this->ESS = _ESS;
	this->Delta = _Delta;
	this->RATE = _RATE;
	this->KLD = _KLD;
    }

    RATEd(std::vector<double> _log_KLD) {
	std::transform(_log_KLD.begin(), _log_KLD.end(), std::back_inserter(this->KLD), static_cast<double(*)(double)>(std::exp));
	this->RATE = rate_from_log_kld(_log_KLD);
	this->Delta = rate_delta(RATE);
	this->ESS = delta_to_ess(Delta);
    }

    RATEd(std::vector<double> _KLD, std::vector<double> _RATE) {
	this->KLD = _KLD;
	this->RATE = _RATE;
	this->Delta = rate_delta(RATE);
	this->ESS = delta_to_ess(Delta);
    }
};

#endif
