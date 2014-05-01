/* bifurcation_chaser.cpp
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


void geteigenvalues(gsl_vector *eigenvalues, unsigned n, const gsl_vector *z, double c) {
	gsl_matrix *hess;
	hess = gsl_matrix_alloc(3 * n + 3, 3 * n + 3);

	domain_energy_fixedHessian(hess, n, z, c);

	domain_eigen_values(eigenvalues, 3 * n + 3, 2 * n, hess);
	gsl_matrix_free(hess);
}


int domain_eigen_perturb(gsl_vector *z, unsigned k, unsigned n,
		unsigned eigen_num, double a0,double a_fact, double c, double eps,
		double g, double N, double energy_thres) {

	printf("Beginning perturbation.\n");

	double a, temp_eigenval, eigenval;
	unsigned kk;

	gsl_vector *temp_z, *eigenvalues, *eigenvector;
	gsl_permutation *eigenorder;
	gsl_matrix *hess;

	eigenvalues = gsl_vector_alloc(3 * n + 3);
	eigenvector = gsl_vector_alloc(3 * n + 3);
	temp_z = gsl_vector_alloc(3 * n + 3);
	hess = gsl_matrix_alloc(3 * n + 3, 3 * n + 3);

	eigenorder = gsl_permutation_alloc(3 * n + 3);

	domain_energy_fixedHessian(hess, n, z, c);

	domain_eigen_values(eigenvalues, 3 * n + 3, 2 * n, hess);
	domain_eigen_sort(eigenorder, 3 * n + 3, eigen_num, eigenvalues);

	kk = gsl_permutation_get(eigenorder, k);

	eigenval = gsl_vector_get(eigenvalues, kk);

	printf("Getting eigenvector.\n");
	domain_energy_fixedHessian(hess, n, z, c);
	domain_eigen_vector(eigenvector, 3 * n + 3, 2 * n, kk, hess);

	a = a0;
	int failed = 0;

	printf("Starting loop.\n");
	while (true) {
		gsl_vector_memcpy(temp_z, z);
		gsl_blas_daxpy(a, eigenvector, temp_z);
		failed = domain_minimize_fixed(z, n, c, eps, N, 0.9, g, 0.9);

		if (failed) {
			printf("Relaxation failed, reducing perturb size.\n");
			a *= 0.1;
		} else {

	domain_energy_fixedHessian(hess, n, z, c);

	domain_eigen_values(eigenvalues, 3 * n + 3, 2 * n, hess);
		domain_eigen_sort(eigenorder,  3 * n + 3, eigen_num, eigenvalues);

		kk = gsl_permutation_get(eigenorder, k);

		temp_eigenval = gsl_vector_get(eigenvalues, kk);

		printf("BIFUR: Perturbing %i, %e, %e\n", k, eigenval, temp_eigenval);

		if (GSL_SIGN(temp_eigenval) != GSL_SIGN(eigenval)) {
			gsl_vector_memcpy(z, temp_z);
			break;
		}

		a *= a_fact;
		}
	}

	gsl_vector_free(eigenvector);
	gsl_vector_free(temp_z);
	gsl_vector_free(eigenvalues);

	gsl_permutation_free(eigenorder);

	return 0;
}


bool bifur_consent() {
	printf(" (y/n): ");
	char in;
	in = getchar();
	getchar();
	if (in == 'y') return true;
	else return false;
}


// Initializes the program.
int main(int argc, char *argv[]) {

	int opt, min_fails, eigen_follow, eigen_num, examining;
	unsigned n, N, j, a, last_pert, ii, old_ii;
	double c, dc0, dc, g0, g, eigen_thres, approach_thres, eps, state, old_state;
	char *filename, str[19], in;
	bool subcrit, reset;

	// Setting default values.
	eps = 0;
	eigen_thres = 1e-13;
	approach_thres = 1e-6;
	eigen_follow = -1;
	examining = -1;
	eigen_num = 25;
	last_pert = 0;
	subcrit = false;
	reset = false;
	dc = 0;

	j = 0;

	gsl_vector *z, *old_z, *eigenvalues, *eigenstate, *old_eigenstate, *eigenchanges;
	gsl_permutation *eigenorder, *old_eigenorder;

	while ((opt = getopt(argc, argv, "n:c:d:g:h:i:N:p:m:j:e:t:s")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'N':
				N = atoi(optarg);
				break;
			case 'j':
				j = atoi(optarg);
				break;
			case 'c':
				c = atof(optarg);
				break;
			case 'd':
				dc0 = atof(optarg);
				break;
			case 'h':
				dc = atof(optarg);
				break;
			case 'g':
				g0 = atof(optarg);
				break;
			case 'i':
				filename = optarg;
				break;
			case 'm':
				eigen_follow = atof(optarg);
				break;
			case 'e':
				eps = atof(optarg);
				break;
			case 's':
				subcrit = true;
				break;
			case 't':
				approach_thres = atof(optarg);
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}

	z = gsl_vector_alloc(3 * n + 3);
	old_z = gsl_vector_alloc(3 * n + 3);
	eigenvalues = gsl_vector_alloc(3 * n + 3);
	eigenorder = gsl_permutation_alloc(3 * n + 3);
	old_eigenorder = gsl_permutation_alloc(3 * n + 3);
	old_eigenstate = gsl_vector_alloc(3 * n + 3);
	eigenstate = gsl_vector_alloc(3 * n + 3);

	FILE *f = fopen(filename, "r");
	gsl_vector_fscanf(f, z);
	fclose(f);

	g = g0;
	if (dc == 0) dc = dc0;

	min_fails = domain_minimize_fixed(z, n, c, eps, N, 0.9, 1, 0.9);

	if (min_fails) {
		printf("BIFUR: Initial relaxation failed, exiting.\n");
		return 1;
	}

	geteigenvalues(eigenvalues, n, z, c);
	domain_eigen_state(old_eigenstate, eigenvalues, n, eigen_thres);
	domain_eigen_sort(old_eigenorder, 3 * n + 3, eigen_num, eigenvalues);

	while (true) {
		j += 1;
		c += dc;
		reset = false;

		gsl_vector_memcpy(old_z, z);

		printf("BIFUR: Step %05d, starting with c = %f.\n", j, c);

		min_fails = domain_minimize_fixed(z, n, c, eps, N, 0.9, 1, 0.9);

		if (min_fails) {
			printf("BIFUR: Newton's method failed to converge, reducing step size.\n");
			c -= dc;
			j -= 1;
			last_pert = 0;
			gsl_vector_memcpy(z, old_z);
			dc *= 0.1;
			reset = true;
		} else {

		geteigenvalues(eigenvalues, n, z, c);
		domain_eigen_sort(eigenorder, 3 * n + 3, eigen_num, eigenvalues);
		domain_eigen_state(eigenstate, eigenvalues, n, eigen_thres);

		if (eigen_follow > -1) examining = eigen_follow;

		for (unsigned i = 0; i < eigen_num; i++) {
			ii = gsl_permutation_get(eigenorder, i);
			old_ii = gsl_permutation_get(old_eigenorder, i);

			state = gsl_vector_get(eigenstate, ii);
			old_state = gsl_vector_get(old_eigenstate, old_ii);

			if (state != old_state) {
				if (i == examining) {
					c -= dc;
					gsl_vector_memcpy(z, old_z);
					gsl_vector_memcpy(eigenstate, old_eigenstate);
					gsl_permutation_memcpy(eigenorder, old_eigenorder);
					j -= 1;
					dc *= 0.1;
					reset = true;
					last_pert = 0;
				} if (examining == -1 && state != 0 && old_state != 0) {
					printf("BIFUR: Eigenvalue %i changed sign past threshold to %e.  Examine?", i,
						gsl_vector_get(eigenvalues, ii));
					if (bifur_consent()) {

						examining = i;
						c -= dc;
						gsl_vector_memcpy(z, old_z);
						gsl_vector_memcpy(eigenstate, old_eigenstate);
						gsl_permutation_memcpy(eigenorder, old_eigenorder);
						j -= 1;
						dc *= 0.1;
						reset = true;
						last_pert = 0;

						break;
					}
				}
			}
		}

		if (!reset && examining > -1 && fabs(dc) < approach_thres) {

			if (!subcrit) {
				c += GSL_SIGN(dc) * approach_thres;
				domain_minimize_fixed(z, n, c, eps, N, 0.9, 1, 0.9);
			}

			printf("BIFUR: Perturbing at c = %.8f.\n", c);

			domain_eigen_perturb(z, examining, n, eigen_num, 1, 1.1, c, eps, g, N, 0);
			geteigenvalues(eigenvalues, n, z, c);
			domain_eigen_sort(eigenorder, 3 * n + 3, eigen_num,  eigenvalues);
			domain_eigen_state(eigenstate, eigenvalues, n, eigen_thres);

			if (subcrit) dc = - GSL_SIGN(dc) * approach_thres;
			else dc = GSL_SIGN(dc) * approach_thres;

			examining = -1;
			last_pert = 0;
		}

		if (!reset) {

			gsl_vector_memcpy(old_eigenstate, eigenstate);
			gsl_permutation_memcpy(old_eigenorder, eigenorder);

			if (last_pert > 10 && fabs(dc) < fabs(dc0)) {
				last_pert = 0;
				dc = GSL_SIGN(dc) * fmin(fabs(dc) * 10, fabs(dc0));
			}

			last_pert += 1;

			double energy = domain_energy_fixedEnergy(n, z, c);

			sprintf(str, "output/out-%05d.dat", j);
			FILE *fout = fopen(str, "w");
			fprintf(fout, "%.10e\t%.10e\n", c, energy);
			for (unsigned i = 0; i < eigen_num; i++) {
				ii = gsl_permutation_get(eigenorder, i);

				fprintf(fout, "%.10e\t", gsl_vector_get(eigenvalues, ii));
			}
			fprintf(fout, "\n");
			gsl_vector_fprintf(fout, z, "%.10e");
			fclose(fout);
		}
	}
	}

	gsl_vector_free(z);

}

