/* domain_eigen.cpp
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

/* A set of utilities for find the generalized eigenvalues and eigenvectors of
 * modulated domains.
 */

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


// Finds the generalized eigenvalues of the Hessian for the state vector z when
// Lambda = c.
void domain_eigen_values(gsl_vector *eigenvalues, unsigned size, unsigned params, gsl_matrix *hess) {

	double eigenvalue;

	gsl_vector *beta;
	gsl_vector_complex *alpha;
	gsl_matrix *modI;
	gsl_eigen_gen_workspace *w;

	alpha = gsl_vector_complex_alloc(size);
	beta = gsl_vector_alloc(size);
	modI = gsl_matrix_alloc(size, size);
	w = gsl_eigen_gen_alloc(size);

	gsl_matrix_set_zero(modI);
	for (unsigned i = 0; i < params; i++) gsl_matrix_set(modI, i, i, 1);

	gsl_eigen_gen(hess, modI, alpha, beta, w);

	for (unsigned i = 0; i < size; i++) {
		eigenvalue = gsl_vector_complex_get(alpha, i).dat[0] / gsl_vector_get(beta, i);
		gsl_vector_set(eigenvalues, i, eigenvalue);
	}

	gsl_vector_free(beta);
	gsl_vector_complex_free(alpha);
	gsl_matrix_free(modI);
	gsl_eigen_gen_free(w);
}


void domain_eigen_sort(gsl_permutation *eigenorder, unsigned size, unsigned eigen_num,
		const gsl_vector *eigenvalues) {

	unsigned ii;

	gsl_vector *abs_eigenvalues;

	abs_eigenvalues = gsl_vector_alloc(size);
	
	for (unsigned i = 0; i < size; i++) {
		gsl_vector_set(abs_eigenvalues, i, fabs(gsl_vector_get(eigenvalues, i)));
	}

	gsl_sort_vector_index(eigenorder, abs_eigenvalues);

	gsl_vector_memcpy(abs_eigenvalues, eigenvalues);

	for (unsigned i = eigen_num; i < size; i++) {
		ii = gsl_permutation_get(eigenorder, i);

		gsl_vector_set(abs_eigenvalues, ii, INFINITY);
	}

	gsl_sort_vector_index(eigenorder, abs_eigenvalues);

	gsl_vector_free(abs_eigenvalues);
}


void domain_eigen_state(gsl_vector *eigenstate, const gsl_vector *eigenvalues,
		unsigned n, double thres) {

	double eigenvalue;

	gsl_vector_set_zero(eigenstate);

	for (unsigned i = 0; i < 3 * n + 3; i++) {
		eigenvalue = gsl_vector_get(eigenvalues, i);
		if (eigenvalue > fabs(thres)) gsl_vector_set(eigenstate, i, 1);
		if (eigenvalue < -fabs(thres)) gsl_vector_set(eigenstate, i, -1);
	}
}


void domain_eigen_vector(gsl_vector *eigenvector, unsigned size, unsigned params, unsigned k, gsl_matrix *hess) {

	gsl_vector *beta;
	gsl_vector_complex *alpha;
	gsl_matrix *modI;
	gsl_matrix_complex *evec;
	gsl_eigen_genv_workspace *w;

	alpha = gsl_vector_complex_alloc(size);
	beta = gsl_vector_alloc(size);
	modI = gsl_matrix_alloc(size, size);
	evec = gsl_matrix_complex_alloc(size, size);
	w = gsl_eigen_genv_alloc(size);

	gsl_matrix_set_zero(modI);

	for (unsigned i = 0; i < params; i++) gsl_matrix_set(modI, i, i, 1);

	gsl_eigen_genv(hess, modI, alpha, beta, evec, w);

	for (unsigned i = 0; i < size; i++) {
		gsl_vector_set(eigenvector, i,
			gsl_matrix_complex_get(evec, i, k).dat[0]);
	}

	gsl_vector_free(beta);
	gsl_vector_complex_free(alpha);
	gsl_matrix_free(modI);
	gsl_matrix_complex_free(evec);
	gsl_eigen_genv_free(w);
}


void domain_eigen_ortho(gsl_vector *eigenvector, unsigned n, const gsl_vector *z) {
	gsl_vector *rotation, *translation_x, *translation_y;
	double x, y, prod;

	rotation = gsl_vector_alloc(3 * n + 3);
	translation_x = gsl_vector_alloc(3 * n + 3);
	translation_y = gsl_vector_alloc(3 * n + 3);

	for (unsigned i = 0; i < n; i++) {
		x = gsl_vector_get(z, i);
		y = 0;

		gsl_vector_set(translation_x, i, 1.0 / n);

		if (n != 0) {
			y = gsl_vector_get(z, n + i - 1);
			gsl_vector_set(translation_y, n + i - 1, 1.0 / n);
		}

		gsl_vector_set(rotation, i, - y / (gsl_pow_2(x) + gsl_pow_2(y)));
		if (n != 0) gsl_vector_set(rotation, n + i - 1, x / (n * (gsl_pow_2(x) + gsl_pow_2(y))));
	}

	gsl_blas_ddot(rotation, eigenvector, &prod);
	prod = prod / gsl_blas_dnrm2(rotation);
	gsl_vector_memcpy(rotation, eigenvector);
	gsl_blas_daxpy(-prod, rotation, eigenvector);

	gsl_blas_ddot(translation_x, eigenvector, &prod);
	prod = prod / gsl_blas_dnrm2(translation_x);
	gsl_vector_memcpy(translation_x, eigenvector);
	gsl_blas_daxpy(-prod, translation_x, eigenvector);

	gsl_blas_ddot(translation_y, eigenvector, &prod);
	prod = prod / gsl_blas_dnrm2(translation_y);
	gsl_vector_memcpy(translation_y, eigenvector);
	gsl_blas_daxpy(-prod, translation_y, eigenvector);

	return;
}


