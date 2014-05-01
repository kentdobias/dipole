/* eigenvalues.cpp
 *
 * Copyright (C) 2013 Jaron Kent-Dobias
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* This program allows for the generalized eigenvalues of a modulated domain to
 * be computed and returned.
 */

#include "domain_energy.h"
#include "domain_eigen.h"

#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>

// GSL includes.
#include <gsl/gsl_sf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort_vector.h>



// Initializes the program.
int main(int argc, char *argv[]) {

	int opt, min_fails, eigen_follow, eigen_num, examining;
	unsigned n, N, ord, size, params, j, M;
	double d, c, dc0, dc, g0, g, eigen_thres, approach_thres, eps, eps2, state, old_state, h, bound;
	char *in_filename, *out_filename, *k_filename, *a_filename, *phi_filename, str[19], in;
	bool subcrit, reset, rand, verbose, fixed;

	// Setting default values.

	gsl_vector *z, *k, *a, *phi, *old_z, *eigenvalues;
	gsl_permutation *eigenorder;
	gsl_matrix *hess;
	rand = false;
	fixed = false;
	verbose = false;
	N=25;

	while ((opt = getopt(argc, argv, "n:c:i:o:O:K:A:P:e:g:N:b:rvd:M:f")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'c':
				c = atof(optarg);
				break;
			case 'i':
				in_filename = optarg;
				break;
			case 'O':
				ord = atoi(optarg);
				break;
			case 'K':
				k_filename = optarg;
				break;
			case 'A':
				a_filename = optarg;
				break;
			case 'P':
				phi_filename = optarg;
				break;
			case 'N':
				N = atoi(optarg);
				break;
			case 'r':
				rand = true;
				break;
			case 'f':
				fixed = true;
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}

	if (rand || !fixed) {
		size = 3 * n + 2;
		params = 2 * n + 1;
	} else {
		size = 3 * n + 3;
		params = 2 * n;
	}

	z = gsl_vector_alloc(size);
	eigenvalues = gsl_vector_alloc(size);
	eigenorder = gsl_permutation_alloc(size);
	hess = gsl_matrix_alloc(size, size);
	if (rand) {
		k = gsl_vector_alloc(2 * ord);
		a = gsl_vector_alloc(ord);
		phi = gsl_vector_alloc(ord);
	}

	FILE *in_file = fopen(in_filename, "r");
	gsl_vector_fscanf(in_file, z);
	fclose(in_file);

	if (rand) {
		FILE *k_file = fopen(k_filename, "r");
		gsl_vector_fscanf(k_file, k);
		fclose(k_file);

		FILE *a_file = fopen(a_filename, "r");
		gsl_vector_fscanf(a_file, a);
		fclose(a_file);

		FILE *phi_file = fopen(phi_filename, "r");
		gsl_vector_fscanf(phi_file, phi);
		fclose(phi_file);
	}

	if (rand) domain_energy_nakedRandHessian(hess, n, z, c, ord, k, a, phi);
	else {
		if (fixed) domain_energy_fixedHessian(hess, n, z, c);
		else domain_energy_nakedHessian(hess, n, z, c);
	}

	domain_eigen_values(eigenvalues, size, params, hess);
	domain_eigen_sort(eigenorder, size, N, eigenvalues);

	for (unsigned i = 0; i < N; i++) {
		printf("%e\t", gsl_vector_get(eigenvalues, gsl_permutation_get(eigenorder, i)));
	}
	printf("\n");

	gsl_vector_free(z);


}
