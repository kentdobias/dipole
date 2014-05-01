/* evolve.cpp
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

/* This program allows for the iterated minimization of modulated domains as
 * the dimensionless parameter Lambda is varied.
 */

#include "domain_energy.h"
#include "domain_minimize.h"

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
	double d, c, dc0, dc, g0, g, eigen_thres, approach_thres, eps, eps2, state, old_state, h, bound, da, w, ss;
	char *in_filename, *out_filename, *k_filename, *a_filename, *phi_filename, str[19], in;
	bool subcrit, reset, rand, verbose, fixed, well;

	// Setting default values.

	gsl_vector *z, *k, *a, *phi, *old_z;
	rand = false;
	fixed = false;
	well = false;
	verbose = false;
	j=0;
	ss=1;

	while ((opt = getopt(argc, argv, "n:c:i:o:O:K:A:P:e:g:N:b:rvd:M:a:fws:W:j:")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'b':
				bound = atof(optarg);
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
			case 'g':
				g0 = atof(optarg);
				break;
			case 'N':
				N = atoi(optarg);
				break;
			case 'j':
				j = atoi(optarg);
				break;
			case 'M':
				M = atoi(optarg);
				break;
			case 'e':
				eps = atof(optarg);
				break;
			case 'd':
				dc0 = atof(optarg);
				break;
			case 'a':
				da = atof(optarg);
				break;
			case 'r':
				rand = true;
				break;
			case 'f':
				fixed = true;
				break;
			case 'w':
				well = true;
				break;
			case 'W':
				w = atof(optarg);
				break;
			case 's':
				ss = atof(optarg);
				break;
			case 'v':
				verbose = true;
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
	old_z = gsl_vector_alloc(size);
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

	g = g0;
	dc = dc0;

	double beta = 0.9;
	double s = 1;
	double sigma = 0.5;

	if (rand && well) min_fails = domain_minimize_randWell(z, n, c, ord, k, a, phi, w, ss, eps, N, beta, s, sigma, g, bound, verbose);
	else if (rand) min_fails = domain_minimize_rand(z, n, c, ord, k, a, phi, eps, N, beta, s, sigma, g, bound, verbose);
	else {
		if (fixed) min_fails = domain_minimize_fixedmin(z, n, c, eps, N, beta, ss, sigma, g, bound, verbose);
		else {
			if (well) min_fails = domain_minimize_nakedWell(z, n, c, w, ss, eps, N, beta, s, sigma, g, bound, verbose);
			else min_fails = domain_minimize_naked(z, n, c, eps, N, beta, s, sigma, g, bound, verbose);
		}
	}

	if (min_fails) {
		printf("BIFUR: Initial relaxation failed, exiting.\n");
		FILE *out_file = fopen(out_filename, "w");
		gsl_vector_fprintf(out_file, z, "%.10e");
		fclose(out_file);
		return 1;
	}


	while (j < M) {
		j += 1;
		c += dc;
		g = g0;
		if (rand) gsl_vector_scale(a, da);

		gsl_vector_memcpy(old_z, z);

		printf("EVOLVE: Step %05d, starting with c = %f.\n", j, c);

		while (true) {
			if (rand && well) min_fails = domain_minimize_randWell(z, n, c, ord, k, a, phi, w, ss, eps, N, beta, s, sigma, g, bound, verbose);
			else if (rand) min_fails = domain_minimize_rand(z, n, c, ord, k, a, phi, eps, N, beta, s, sigma, g, bound, verbose);
			else if (fixed) min_fails = domain_minimize_fixedmin(z, n, c, eps, N, beta, s, sigma, g, bound, verbose);
			else if (well) min_fails = domain_minimize_nakedWell(z, n, c, w, ss, eps, N, beta, s, sigma, g, bound, verbose);
			else min_fails = domain_minimize_naked(z, n, c, eps, N, beta, s, sigma, g, bound, verbose);

			if (!min_fails) break;
			printf("EVOLVE: Newton's method failed to converge, reducing gamma.\n");
			gsl_vector_memcpy(z, old_z);
			g *= 0.1;
		}

		sprintf(str, "output/out-%05d.dat", j);
		FILE *fout = fopen(str, "w");
		fprintf(fout, "%.10e\n", c);
		gsl_vector_fprintf(fout, z, "%.10e");
		fclose(fout);
	}

	FILE *out_file = fopen(out_filename, "w");
	gsl_vector_fprintf(out_file, z, "%.10e");
	fclose(out_file);

	gsl_vector_free(z);

	return 0;

}
