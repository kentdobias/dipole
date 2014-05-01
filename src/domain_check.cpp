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

void bifur_eigenvalues(gsl_vector *eigenvalues, unsigned n,
		const gsl_vector *z, double c) {

	double eigenvalue;

	gsl_vector *beta;
	gsl_vector_complex *alpha;
	gsl_matrix *hess, *modI;
	gsl_eigen_gen_workspace *w;

	alpha = gsl_vector_complex_alloc(3 * n + 3);
	beta = gsl_vector_alloc(3 * n + 3);
	hess = gsl_matrix_alloc(3 * n + 3, 3 * n + 3);
	modI = gsl_matrix_alloc(3 * n + 3, 3 * n + 3);
	w = gsl_eigen_gen_alloc(3 * n + 3);

	gsl_matrix_set_zero(modI);
	for (unsigned i = 0; i < 2 * n; i++) gsl_matrix_set(modI, i, i, 1);

	domain_energy_hessian(hess, n, z, c);

	gsl_eigen_gen(hess, modI, alpha, beta, w);

	for (unsigned i = 0; i < 3 * n + 3; i++) {
		eigenvalue = gsl_vector_complex_get(alpha, i).dat[0] / gsl_vector_get(beta, i);
		gsl_vector_set(eigenvalues, i, eigenvalue);
	}

	gsl_vector_free(beta);
	gsl_vector_complex_free(alpha);
	gsl_matrix_free(modI);
	gsl_matrix_free(hess);
	gsl_eigen_gen_free(w);
}

void bifur_trueEigenvalues(gsl_vector *eigenvalues, unsigned n,
		const gsl_vector *z, double c) {

	double eigenvalue;

	gsl_vector *beta;
	gsl_vector_complex *alpha;
	gsl_matrix *hess, *modI;
	gsl_eigen_gen_workspace *w;

	alpha = gsl_vector_complex_alloc(3 * n + 4);
	beta = gsl_vector_alloc(3 * n + 4);
	hess = gsl_matrix_alloc(3 * n + 4, 3 * n + 4);
	modI = gsl_matrix_alloc(3 * n + 4, 3 * n + 4);
	w = gsl_eigen_gen_alloc(3 * n + 4);

	gsl_matrix_set_zero(modI);
	for (unsigned i = 0; i < 2 * n + 1; i++) gsl_matrix_set(modI, i, i, 1);

	domain_energy_truehessian(hess, n, z, c);

	gsl_eigen_gen(hess, modI, alpha, beta, w);

	for (unsigned i = 0; i < 3 * n + 4; i++) {
		eigenvalue = gsl_vector_complex_get(alpha, i).dat[0] / gsl_vector_get(beta, i);
		gsl_vector_set(eigenvalues, i, eigenvalue);
	}

	gsl_vector_free(beta);
	gsl_vector_complex_free(alpha);
	gsl_matrix_free(modI);
	gsl_matrix_free(hess);
	gsl_eigen_gen_free(w);
}


void bifur_eigensort(gsl_permutation *eigenorder, unsigned n, unsigned eigen_num,
		const gsl_vector *eigenvalues) {

	unsigned ii;

	gsl_vector *abs_eigenvalues;

	abs_eigenvalues = gsl_vector_alloc(3 * n + 3);
	
	for (unsigned i = 0; i < 3 * n + 3; i++) {
		gsl_vector_set(abs_eigenvalues, i, fabs(gsl_vector_get(eigenvalues, i)));
	}

	gsl_sort_vector_index(eigenorder, abs_eigenvalues);

	gsl_vector_memcpy(abs_eigenvalues, eigenvalues);

	for (unsigned i = eigen_num; i < 3 * n + 3; i++) {
		ii = gsl_permutation_get(eigenorder, i);

		gsl_vector_set(abs_eigenvalues, ii, INFINITY);
	}

	gsl_sort_vector_index(eigenorder, abs_eigenvalues);

	gsl_vector_free(abs_eigenvalues);
}

void bifur_trueEigensort(gsl_permutation *eigenorder, unsigned n, unsigned eigen_num,
		const gsl_vector *eigenvalues) {

	unsigned ii;

	gsl_vector *abs_eigenvalues;

	abs_eigenvalues = gsl_vector_alloc(3 * n + 4);
	
	for (unsigned i = 0; i < 3 * n + 4; i++) {
		gsl_vector_set(abs_eigenvalues, i, fabs(gsl_vector_get(eigenvalues, i)));
	}

	gsl_sort_vector_index(eigenorder, abs_eigenvalues);

	gsl_vector_memcpy(abs_eigenvalues, eigenvalues);

	for (unsigned i = eigen_num; i < 3 * n + 4; i++) {
		ii = gsl_permutation_get(eigenorder, i);

		gsl_vector_set(abs_eigenvalues, ii, INFINITY);
	}

	gsl_sort_vector_index(eigenorder, abs_eigenvalues);

	gsl_vector_free(abs_eigenvalues);
}



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

	min_fails = domain_minimize(z, n, c, eps, g, N, 4, 2, 0.9);

	if (min_fails) {
		printf("BIFUR: Initial relaxation failed, exiting.\n");
		return 1;
	}

	bifur_eigenvalues(eigenvalues, n, z, c);
	bifur_eigensort(eigenorder, n, num, eigenvalues);
	bifur_trueEigenvalues(trueEigenvalues, n, z, c);
	bifur_trueEigensort(trueEigenorder, n, num, trueEigenvalues);

	energy = domain_energy_energy(n, z, c);
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
