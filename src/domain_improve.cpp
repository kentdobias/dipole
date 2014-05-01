/* domain_improve.cpp
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

/* A program which facilitates automated mapping of bifurcation points in the
 * energy of a system where the Hessian is available.  Currently, only a one
 * dimensional parameter space is supported.
 */

#include "domain_energy.h"
#include "domain_minimize.h"
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

	int opt, min_fails;
	unsigned n, N, num;
	double c, g0, g, eps, energy;
	char *filename;
	bool eigenpres = true;

	// Setting default values.
	eps = 0;
	num = 25;

	gsl_vector *z, *old_z, *eigenvalues, *trueEigenvalues;
	gsl_permutation *eigenorder, *trueEigenorder;

	while ((opt = getopt(argc, argv, "n:c:d:g:h:i:N:p:m:j:e:t:s")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'N':
				N = atoi(optarg);
				break;
			case 'g':
				g0 = atof(optarg);
				break;
			case 'i':
				filename = optarg;
				break;
			case 'e':
				eps = atof(optarg);
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}

	z = gsl_vector_alloc(3 * n + 3);
	old_z = gsl_vector_alloc(3 * n + 3);
	eigenvalues = gsl_vector_alloc(3 * n + 3);
	trueEigenvalues = gsl_vector_alloc(3 * n + 4);
	eigenorder = gsl_permutation_alloc(3 * n + 3);
	trueEigenorder = gsl_permutation_alloc(3 * n + 4);

	g = g0;

	char ch;
	double throwaway;

	FILE *f = fopen(filename, "r+");
	while (ch != '\n') ch = fgetc(f);
	ch = 'a';
	while (ch != '\n' && ch != '\t') ch = fgetc(f);
	if (ch == '\n') eigenpres = false;

	rewind(f);

	fscanf(f, "%le\t", &c);
	fscanf(f, "%le\n", &energy);

	if (eigenpres) {
		ch = 'a';
		while (ch != '\n') ch = fgetc(f);
	}
	gsl_vector_fscanf(f, z);
	fclose(f);

	min_fails = domain_minimize_fixed(z, n, c, eps, g, N, 4, 2);

	if (min_fails) {
		printf("BIFUR: Initial relaxation failed, exiting.\n");
		return 1;
	}

//	domain_eigen_values(eigenvalues, n, z, c);
//	domain_eigen_sort(eigenorder, n, num, eigenvalues);

	energy = domain_energy_fixedEnergy(n, z, c);
	unsigned ii;

	FILE *newf = fopen(filename, "w");
	fprintf(newf, "%.12le\t%.12le\n", c, energy);
	for (unsigned i = 0; i < num; i++) {
		ii = gsl_permutation_get(eigenorder, i);
		fprintf(newf, "%.12le\t", gsl_vector_get(eigenvalues, ii));
	}
	fprintf(newf, "\n");
	for (unsigned i = 0; i < num; i++) {
		ii = gsl_permutation_get(trueEigenorder, i);
		fprintf(newf, "%.12le\t", gsl_vector_get(trueEigenvalues, ii));
	}
	fprintf(newf, "\n");
	for (unsigned i = 0; i < 3 * n + 3; i++) {
		fprintf(newf, "%.12le\t", gsl_vector_get(z, i));
	}
	fclose(newf);
}
