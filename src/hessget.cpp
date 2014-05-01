/* eigenget.cpp
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

// A utility which returns a given generalized eigenvector to the user.

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

	int opt;
	unsigned n, m, nn;
	double c;
	char *in_filename, *out_filename;
	bool truehessian;

	gsl_vector *z;
	gsl_matrix *hess, *testhess;

	truehessian = false;

	while ((opt = getopt(argc, argv, "n:m:c:i:o:t")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'm':
				m = atoi(optarg);
				break;
			case 'c':
				c = atof(optarg);
				break;
			case 'i':
				in_filename = optarg;
				break;
			case 'o':
				out_filename = optarg;
				break;
			case 't':
				truehessian = true;
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}

	if (truehessian) nn = 3 * n + 4;
	else nn = 3 * n + 3;

	z = gsl_vector_alloc(3 * n + 3);
	hess = gsl_matrix_alloc(nn, nn);
	testhess = gsl_matrix_alloc(nn, nn);

	FILE *in_file = fopen(in_filename, "r");
	gsl_vector_fscanf(in_file, z);
	fclose(in_file);

	domain_energy_fixedHessian(hess, n, z, c);
	domain_energy_fixedHessian(testhess, n, z, c);

	FILE *out_file = fopen(out_filename, "w");
	for (unsigned i = 0; i < nn; i++) {
		for (unsigned j = 0; j < nn; j++) {
			fprintf(out_file, "%e\t", gsl_matrix_get(hess, i, j) - gsl_matrix_get(testhess, i, j));
		}
		fprintf(out_file, "\n");
	}
	fclose(out_file);
}


