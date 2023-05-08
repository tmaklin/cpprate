// rcgpar: parallel estimation of mixture model components
// https://github.com/tmaklin/rcgpar
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
// USA
//
#ifndef RCGPAR_MPIHANDLER_HPP
#define RCGPAR_MPIHANDLER_HPP

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
