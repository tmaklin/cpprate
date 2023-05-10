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
#ifndef CPPRATE_MPIHANDLER_HPP
#define CPPRATE_MPIHANDLER_HPP

#include "cpprate_mpi_config.hpp"

#include <cstddef>

class MpiHandler {
    // Class that takes care of dividing a n x m matrix to the MPI
    // tasks and gathering the results from the tasks back together.
private:
    int n_tasks;
    int rank;
    int status;

    int displacements[1024];
    int bufcounts[1024] = { 0 };

public:
    MpiHandler() {
	this->status = MPI_Comm_size(MPI_COMM_WORLD, &this->n_tasks);
	this->status = MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    }
    uint32_t obs_per_task(const uint32_t n_obs) const {
	uint32_t n_obs_per_task = std::floor(n_obs/this->n_tasks);
	if (rank == (this->n_tasks - 1)) {
	    // Last process takes care of observations assigned to remainder.
	    n_obs_per_task += n_obs - n_obs_per_task*this->n_tasks;
	}
	return n_obs_per_task;
    }
    void initialize_1(const uint32_t n_obs) {
	// Initializes the displacements and bufcounts.
	uint32_t sent_so_far = 0;
	uint32_t n_obs_per_task = std::floor(n_obs/this->n_tasks);
	for (uint16_t i = 0; i < this->n_tasks - 1; ++i) {
	    this->displacements[i] = sent_so_far;
	    this->bufcounts[i] = n_obs*n_obs_per_task;
	    sent_so_far += this->bufcounts[i];
	}
	this->displacements[this->n_tasks - 1] = sent_so_far;
	this->bufcounts[this->n_tasks - 1] = n_obs*n_obs - sent_so_far;
    }

    void initialize_2(const uint32_t n_obs) {
	// Initializes the displacements and bufcounts.
	uint32_t sent_so_far = 0;
	uint32_t n_obs_per_task = std::floor(n_obs/this->n_tasks);
	for (uint16_t i = 0; i < this->n_tasks - 1; ++i) {
	    this->displacements[i] = sent_so_far;
	    this->bufcounts[i] = n_obs_per_task;
	    sent_so_far += this->bufcounts[i];
	}
	this->displacements[this->n_tasks - 1] = sent_so_far;
	this->bufcounts[this->n_tasks - 1] = n_obs - sent_so_far;
    }

    const int* get_displacements() const {
	return this->displacements;
    }
    const int* get_bufcounts() const {
	return this->bufcounts;
    }
    int get_status() const {
	return this->status;
    }
    int get_n_tasks() const {
	return this->n_tasks;
    }
    int get_rank() const {
	return this->rank;
    }
};

#endif
