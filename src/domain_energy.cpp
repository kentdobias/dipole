/* domain_energy.cpp
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

/* A lightweight and efficient model of two dimensional phase-modulated domains
 * using Newton's method to minimize a Lagrangian.
 *
 * Best viewed in an 80 character terminal with two character hard tabs.
 */


#include <stdlib.h>
#include <math.h>

// GSL includes.
#include <gsl/gsl_sf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_blas.h>


void domain_energy_x(gsl_vector *x, unsigned n, const gsl_vector *z) {
// Gets the full set of x coordinates from the state array.

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));

}


void domain_energy_y(gsl_vector *y, unsigned n, const gsl_vector *z) {
// Gets the full set of y coordinates from the state array.

	gsl_vector_set(y, 0, 0);

	#pragma omp parallel for
	for (unsigned i = 1; i < n; i++) {
		gsl_vector_set(y, i, gsl_vector_get(z, n + i - 1));
	}

}


void gsl_permutation_over(unsigned n, gsl_permutation *perm, bool right) {
// Shifts a GSL permutation object circularly.  If right is true, then the
// permutation is shifted to the right; if false, then it is shifted to the
// left.

	gsl_permutation_swap(perm, 0, n - 1);

	if (right) {
		for (unsigned i = 0; i < n - 2; i++) {
			gsl_permutation_swap(perm, n - 1 - i, n - 2 - i);
		}
	}

	else {
		for (unsigned i = 0; i < n - 2; i++) {
			gsl_permutation_swap(perm, i, i + 1);
		}
	}
}


double domain_energy_area(unsigned n, const gsl_vector *x, const gsl_vector *y) {
// Computes the area of a domain.

	double area, x_i, y_i, x_ii, y_ii;
	unsigned ii;

	gsl_permutation *indices_left;

	indices_left = gsl_permutation_alloc(n);
	gsl_permutation_init(indices_left);
	gsl_permutation_over(n, indices_left, false);

	area = 0;

	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_left, i);

		x_i = gsl_vector_get(x, i);
		y_i = gsl_vector_get(y, i);
		x_ii = gsl_vector_get(x, ii);
		y_ii = gsl_vector_get(y, ii);

		area += (x_i * y_ii - x_ii * y_i) / 2;
	}

	gsl_permutation_free(indices_left);

	return area;
}


void domain_energy_rt(gsl_vector *rt, unsigned n, const gsl_vector *x,
		double p) {
// Converts x and y coordinates to r_x, r_y, t_x, or t_y, depending on input.

	double x_i, x_ii;
	unsigned ii;

	gsl_permutation *indices_left;

	indices_left = gsl_permutation_alloc(n);
	gsl_permutation_init(indices_left);
	gsl_permutation_over(n, indices_left, false);

	#pragma omp parallel for private(ii, x_i, x_ii)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_left, i);

		x_i = gsl_vector_get(x, i);
		x_ii = gsl_vector_get(x, ii);

		gsl_vector_set(rt, i, x_ii + p * x_i);
	}

	gsl_permutation_free(indices_left);
}


void domain_energy_tdots(gsl_matrix *tdots, unsigned n, const gsl_vector *tx,
		const gsl_vector *ty) {
// Creates a matrix of dotted tangent vectors.

	gsl_matrix_set_zero(tdots);

	gsl_blas_dger(1, tx, tx, tdots);
	gsl_blas_dger(1, ty, ty, tdots);
}


void domain_energy_dists(gsl_matrix *dists, unsigned n, const gsl_vector *rx,
		const gsl_vector *ry) {
// Creates a matrix of distances between points on the domain boundary..

	double rx_i, rx_j, ry_i, ry_j;

	#pragma omp parallel for private(rx_i, rx_j, ry_i, ry_j)
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < n; j++) {
			rx_i = gsl_vector_get(rx, i);
			rx_j = gsl_vector_get(rx, j);

			ry_i = gsl_vector_get(ry, i);
			ry_j = gsl_vector_get(ry, j);

			gsl_matrix_set(dists, i, j,
					2 / sqrt(gsl_pow_2(rx_i - rx_j) + gsl_pow_2(ry_i - ry_j)));
		}
	}

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		gsl_matrix_set(dists, i, i, 0);
	}
}


double domain_energy_dconst(unsigned n, double L, double *ld,
		const gsl_vector *tx, const gsl_vector *ty) {
// Computes the value of the Lagrangian constraint on the distances.

	double dconst, tx_i, ty_i;

	dconst = 0;

	for (unsigned i = 0; i < n; i++) {
		tx_i = gsl_vector_get(tx, i);
		ty_i = gsl_vector_get(ty, i);

		dconst += ld[i] * (gsl_pow_2(L / n)
			- (gsl_pow_2(tx_i) + gsl_pow_2(ty_i)));
	}

	return dconst;
}


double domain_energy_dval(unsigned n, double L, double *ld,
		const gsl_vector *tx, const gsl_vector *ty) {
// Computes the value of the Lagrangian constraint on the distances.

	double dconst, tx_i, ty_i;

	dconst = 0;

	for (unsigned i = 0; i < n; i++) {
		tx_i = gsl_vector_get(tx, i);
		ty_i = gsl_vector_get(ty, i);

		dconst += gsl_pow_2(gsl_pow_2(L / n)
			- (gsl_pow_2(tx_i) + gsl_pow_2(ty_i)));
	}

	return dconst;
}


double domain_energy_init(unsigned n, const gsl_vector *z, gsl_vector *rx,
		gsl_vector *ry, gsl_vector *tx, gsl_vector *ty, gsl_matrix *tdots,
		gsl_matrix *dists) {
// Get useful objects from the state vector.  Fills all GSL matrix and vector
// objects, and returns the value of the area.

	double area;

	gsl_vector *x, *y;
	
	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	domain_energy_x(x, n, z);
	domain_energy_y(y, n, z);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	area = domain_energy_area(n, x, y);

	gsl_vector_free(x);
	gsl_vector_free(y);

	return area;
}


double domain_energy_energy(unsigned n,
		double c, const gsl_vector *rx, const gsl_vector *ry, const gsl_vector *tx, const gsl_vector *ty, gsl_matrix *tdots, gsl_matrix *dists, double L) {
// Computes the Lagrangian.

	double energy_mu, energy, lagrangian, H;
 
	gsl_vector *v_temp_a, *v_temp_b;

	v_temp_a = gsl_vector_alloc(n);
	v_temp_b = gsl_vector_alloc(n);

	gsl_vector_set_all(v_temp_a, 1);
	gsl_vector_set_all(v_temp_b, 1);

	gsl_matrix_mul_elements(tdots, dists);
	gsl_blas_dtrmv(CblasUpper, CblasNoTrans, CblasNonUnit, tdots, v_temp_a);
	gsl_blas_ddot(v_temp_a, v_temp_b, &energy_mu);

	H = log(0.5 * (n - 1)) +  M_EULER + 1.0 / (n - 1)
		- 1.0 / (12 * gsl_pow_2(0.5 * (n - 1)));

	L = fabs(L);

	energy = c * L - L * log(L) - energy_mu + L * H;

	gsl_vector_free(v_temp_a);
	gsl_vector_free(v_temp_b);

	return energy;
}

double domain_energy_lagrangian(unsigned n,
		double c, const gsl_vector *rx, const gsl_vector *ry, const gsl_vector *tx, const gsl_vector *ty, gsl_matrix *tdots, gsl_matrix *dists, double L, double area, double la, double *ld) {
// Computes the Lagrangian.

	double energy, lagrangian;

	energy = domain_energy_energy(n, c, rx, ry, tx, ty, tdots, dists, L);

	L = fabs(L);

	lagrangian = energy - la * (area - M_PI) - domain_energy_dconst(n, L, ld, tx, ty);

	return lagrangian;
}


void domain_energy_gradient(gsl_vector *grad, unsigned n,
		double c, const gsl_vector *rx, const gsl_vector *ry, const gsl_vector *tx, const gsl_vector *ty, const gsl_matrix *tdots, const gsl_matrix *dists, double L, double area, double la, double *ld) {

// Computes the gradient of the Lagrangian.
	double rx_i, rx_ii, rx_j, tx_i, tx_ii, tx_j, ry_i, ry_ii,
	ry_j, ty_i, ty_ii, ty_j, d_ij, d_iij, tdt_ij, tdt_iij, d_ij3, d_iij3;
	unsigned ii, jj;

	gsl_vector *v_ones, *v_storage;
	gsl_matrix *m_dx, *m_dy;
	gsl_permutation *indices_right;
	gsl_permutation *indices_left;

	v_ones = gsl_vector_alloc(n);
	v_storage = gsl_vector_alloc(n);

	m_dx = gsl_matrix_alloc(n, n);
	m_dy = gsl_matrix_alloc(n, n);

	indices_right = gsl_permutation_alloc(n);
	indices_left = gsl_permutation_alloc(n);

	gsl_vector_set_zero(grad);
	gsl_vector_set_all(v_ones, 1);
	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);
	gsl_permutation_init(indices_left);
	gsl_permutation_over(n, indices_left, false);

	#pragma omp parallel for private(rx_i, rx_j, tx_j, tdt_ij, d_ij, rx_ii, ry_i, ry_j, ty_j, ry_ii, tdt_iij, d_iij, d_ij3, d_iij3, ii)
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < n; j++) {
			ii = gsl_permutation_get(indices_right, i);

			rx_i = gsl_vector_get(rx, i);
			rx_ii = gsl_vector_get(rx, ii);
			rx_j = gsl_vector_get(rx, j);
			tx_j = gsl_vector_get(tx, j);

			ry_i = gsl_vector_get(ry, i);
			ry_ii = gsl_vector_get(ry, ii);
			ry_j = gsl_vector_get(ry, j);
			ty_j = gsl_vector_get(ty, j);

			d_ij = gsl_matrix_get(dists, i, j);
			d_iij = gsl_matrix_get(dists, ii, j);
			tdt_ij = gsl_matrix_get(tdots, i, j);
			tdt_iij = gsl_matrix_get(tdots, ii, j);

			d_ij3 = gsl_pow_3(d_ij);
			d_iij3 = gsl_pow_3(d_iij);

			gsl_matrix_set(m_dx, i, j,
					- tx_j * d_ij - (rx_i - rx_j) * tdt_ij * d_ij3 / 4
					+ tx_j * d_iij - (rx_ii - rx_j) * tdt_iij * d_iij3 / 4);

			gsl_matrix_set(m_dy, i, j,
					- ty_j * d_ij - (ry_i - ry_j) * tdt_ij * d_ij3 / 4
					+ ty_j * d_iij - (ry_ii - ry_j) * tdt_iij * d_iij3 / 4);
		}
	}

	#pragma omp parallel for private(ii)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);
		gsl_matrix_set(m_dx, i, ii, 0);
		gsl_matrix_set(m_dx, i, i, 0);
		gsl_matrix_set(m_dy, i, ii, 0);
		gsl_matrix_set(m_dy, i, i, 0);
	}

	gsl_blas_dgemv(CblasNoTrans, 1, m_dx, v_ones, 0, v_storage);

	#pragma omp parallel for private(ii, tx_i, tx_ii, d_ij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);

		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);
		d_ij = gsl_matrix_get(dists, i, ii);

		gsl_vector_set(grad, i,
				- (gsl_vector_get(v_storage, i) + (tx_i - tx_ii) * d_ij));
	}

	gsl_blas_dgemv(CblasNoTrans, 1, m_dy, v_ones, 0, v_storage);

	#pragma omp parallel for private(ii, ty_i, ty_ii, d_ij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);

		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);
		d_ij = gsl_matrix_get(dists, i, ii);

		gsl_vector_set(grad, i + n,
				- (gsl_vector_get(v_storage, i) + (ty_i - ty_ii) * d_ij));
	}

	// darea/dx_i or y_i

	#pragma omp parallel for private(ii, jj)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);
		jj = gsl_permutation_get(indices_left, i);

		gsl_vector_set(grad, i, gsl_vector_get(grad, i) -
				0.5 * (gsl_vector_get(ty, i) + gsl_vector_get(ty, ii)) * la);

		gsl_vector_set(grad, n + i, gsl_vector_get(grad, n + i) +
				0.5 * (gsl_vector_get(tx, i) + gsl_vector_get(tx, ii)) * la);
	}

	#pragma omp parallel for private(ii, jj, tx_i, tx_ii, ty_i, ty_ii)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);
		jj = gsl_permutation_get(indices_left, i);

		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);

		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);

		gsl_vector_set(grad, i, gsl_vector_get(grad, i) -
				2 * (ld[i] * tx_i - ld[ii] * tx_ii));

		gsl_vector_set(grad, i + n, gsl_vector_get(grad, i + n) -
				2 * (ld[i] * ty_i - ld[ii] * ty_ii));
	}

	// The gradient with respect to L.

	L = fabs(L);
	double gradLDist = 0;
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);
		jj = gsl_permutation_get(indices_left, i);

		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);

		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);

		gradLDist += 2 * L / gsl_pow_2(n) * ld[i];
	}

	double H = log(0.5 * (n - 1)) +  M_EULER + 1.0 / (n - 1)
		- 1.0 / (12 * gsl_pow_2(0.5 * (n - 1)));

	double gradL = GSL_SIGN(L) * (c - (1 + log(L) - H) - gradLDist);

	gsl_vector_set(grad, 2 * n, gradL);

	// The gradients with respect to the undetermined coefficients are simply
	// the constraints.

	double gradla = M_PI - area;

	gsl_vector_set(grad, 2 * n + 1, gradla);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		gsl_vector_set(grad, 2 * n + 2 + i, gsl_pow_2(gsl_vector_get(tx, i)) +
				gsl_pow_2(gsl_vector_get(ty, i)) - gsl_pow_2(L / n));
	}

	gsl_vector_free(v_ones);
	gsl_vector_free(v_storage);

	gsl_matrix_free(m_dx);
	gsl_matrix_free(m_dy);

	gsl_permutation_free(indices_right);
	gsl_permutation_free(indices_left);
}


void domain_energy_halfHessian(gsl_matrix *hess, unsigned n,
		double c, const gsl_vector *rx, const gsl_vector *ry, const gsl_vector *tx, const gsl_vector *ty, const gsl_matrix *tdots, const gsl_matrix *dists, double L, double la, double *ld) {
/* Computes the Hessian of the Lagrangian without the center of mass
 * constraints and fixed point.
 */

	gsl_matrix *m_dxidxj, *m_dyidyj, *m_dxidxi, *m_dyidyi, *m_dxidxii,
						 *m_dyidyii, *m_dxidyj, *m_dxidyi, *m_dxiidyi, *m_dxidyii;
	gsl_vector *v_ones, *v_storage;
	gsl_permutation *indicesRight, *indicesLeft, *indices2Right;

	unsigned ii, jj, iii;
	double rx_i, rx_j, tx_i, tx_j, tdt_ij, d_ij, rx_ii, rx_jj, tx_ii, tx_jj,
				 ry_i, ry_j, ty_i, ty_j, ry_ii, ry_jj, ty_ii, ty_jj, tdt_iij, d_iij,
				 tdt_ijj, d_ijj, tdt_iijj, d_iijj, rx_iii, tx_iii, ry_iii, ty_iii,
				 d_ij3, d_iij3, d_ijj3, d_iijj3, d_ij5, d_iij5, d_ijj5, d_iijj5;

	m_dxidxj = gsl_matrix_alloc(n, n);
	m_dyidyj = gsl_matrix_alloc(n, n);
	m_dxidxi = gsl_matrix_alloc(n, n);
	m_dyidyi = gsl_matrix_alloc(n, n);
	m_dxidxii = gsl_matrix_alloc(n, n);
	m_dyidyii = gsl_matrix_alloc(n, n);
	m_dxidyj = gsl_matrix_alloc(n, n);
	m_dxidyi = gsl_matrix_alloc(n, n);
	m_dxiidyi = gsl_matrix_alloc(n, n);
	m_dxidyii = gsl_matrix_alloc(n, n);

	v_ones = gsl_vector_alloc(n);
	v_storage = gsl_vector_alloc(n);

	indicesRight = gsl_permutation_alloc(n);
	indicesLeft = gsl_permutation_alloc(n);
	indices2Right = gsl_permutation_alloc(n);

	gsl_matrix_set_zero(hess);
	gsl_vector_set_all(v_ones, 1);

	gsl_permutation_init(indicesRight);
	gsl_permutation_init(indicesLeft);
	gsl_permutation_over(n, indicesRight, true);
	gsl_permutation_over(n, indicesLeft, false);
	gsl_permutation_memcpy(indices2Right, indicesRight);
	gsl_permutation_over(n, indices2Right, true);

	#pragma omp parallel for private(rx_i, rx_j, tx_i, tx_j, tdt_ij, d_ij, rx_ii, rx_jj, tx_ii, tx_jj, ry_i, ry_j, ty_i, ty_j, ry_ii, ry_jj, ty_ii, ty_jj, tdt_iij, d_iij, tdt_ijj, d_ijj, tdt_iijj, d_iijj, rx_iii, tx_iii, ry_iii, ty_iii, d_ij3, d_iij3, d_ijj3, d_iijj3, d_ij5, d_iij5, d_ijj5, d_iijj5, ii, jj)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);
		for (unsigned j = 0; j < n; j++) {
			jj = gsl_permutation_get(indicesRight, j);

			rx_i = gsl_vector_get(rx, i);
			rx_j = gsl_vector_get(rx, j);
			tx_i = gsl_vector_get(tx, i);
			tx_j = gsl_vector_get(tx, j);

			ry_i = gsl_vector_get(ry, i);
			ry_j = gsl_vector_get(ry, j);
			ty_i = gsl_vector_get(ty, i);
			ty_j = gsl_vector_get(ty, j);

			rx_ii = gsl_vector_get(rx, ii);
			rx_jj = gsl_vector_get(rx, jj);
			tx_ii = gsl_vector_get(tx, ii);
			tx_jj = gsl_vector_get(tx, jj);

			ry_ii = gsl_vector_get(ry, ii);
			ry_jj = gsl_vector_get(ry, jj);
			ty_ii = gsl_vector_get(ty, ii);
			ty_jj = gsl_vector_get(ty, jj);

			d_ij = gsl_matrix_get(dists, i, j);
			tdt_ij = gsl_matrix_get(tdots, i, j);

			d_iij = gsl_matrix_get(dists, ii, j);
			tdt_iij = gsl_matrix_get(tdots, ii, j);

			d_ijj = gsl_matrix_get(dists, i, jj);
			tdt_ijj = gsl_matrix_get(tdots, i, jj);

			d_iijj = gsl_matrix_get(dists, ii, jj);
			tdt_iijj = gsl_matrix_get(tdots, ii, jj);

			d_ij3 = gsl_pow_3(d_ij);
			d_iij3 = gsl_pow_3(d_iij);
			d_ijj3 = gsl_pow_3(d_ijj);
			d_iijj3 = gsl_pow_3(d_iijj);
			d_ij5 = gsl_pow_5(d_ij);
			d_iij5 = gsl_pow_5(d_iij);
			d_ijj5 = gsl_pow_5(d_ijj);
			d_iijj5 = gsl_pow_5(d_iijj);

			// d^2E / dx_i dx_j for i != j, j - 1.
			gsl_matrix_set(m_dxidxj, i, j,
					(rx_i - rx_j) * (tx_i - tx_j) * d_ij3 / 4 + d_ij
					- 3 * gsl_pow_2(rx_i - rx_j) * tdt_ij * d_ij5 / 16
					+ tdt_ij * d_ij3 / 4

					+ (rx_ii - rx_j) * (tx_ii + tx_j) * d_iij3 / 4 - d_iij
					- 3 * gsl_pow_2(rx_ii - rx_j) * tdt_iij * d_iij5 / 16
					+ tdt_iij * d_iij3 / 4

					- (rx_i - rx_jj) * (tx_i + tx_jj) * d_ijj3 / 4 - d_ijj
					- 3 * gsl_pow_2(rx_i - rx_jj) * tdt_ijj * d_ijj5 / 16
					+ tdt_ijj * d_ijj3 / 4

					- (rx_ii - rx_jj) * (tx_ii - tx_jj) * d_iijj3 / 4 + d_iijj
					- 3 * gsl_pow_2(rx_ii - rx_jj) * tdt_iijj * d_iijj5 / 16
					+ tdt_iijj * d_iijj3 / 4
					);

			// d^2E / dy_i dy_j for i != j, j - 1.
			gsl_matrix_set(m_dyidyj, i, j,
					(ry_i - ry_j) * (ty_i - ty_j) * d_ij3 / 4 + d_ij
					- 3 * gsl_pow_2(ry_i - ry_j) * tdt_ij * d_ij5 / 16
					+ tdt_ij * d_ij3 / 4

					+ (ry_ii - ry_j) * (ty_ii + ty_j) * d_iij3 / 4 - d_iij
					- 3 * gsl_pow_2(ry_ii - ry_j) * tdt_iij * d_iij5 / 16
					+ tdt_iij * d_iij3 / 4

					- (ry_i - ry_jj) * (ty_i + ty_jj) * d_ijj3 / 4 - d_ijj
					- 3 * gsl_pow_2(ry_i - ry_jj) * tdt_ijj * d_ijj5 / 16
					+ tdt_ijj * d_ijj3 / 4

					- (ry_ii - ry_jj) * (ty_ii - ty_jj) * d_iijj3 / 4 + d_iijj
					- 3 * gsl_pow_2(ry_ii - ry_jj) * tdt_iijj * d_iijj5 / 16
					+ tdt_iijj * d_iijj3 / 4
					);

			// d^2E / dx_i^2
			gsl_matrix_set(m_dxidxi, i, j,
					(rx_i - rx_j) * tx_j * d_ij3 / 2
					+ 3 * gsl_pow_2(rx_i - rx_j) * tdt_ij * d_ij5 / 16
					- tdt_ij * d_ij3 / 4

					- (rx_ii - rx_j) * tx_j * d_iij3 / 2
					+ 3 * gsl_pow_2(rx_ii - rx_j) * tdt_iij * d_iij5 / 16
					- tdt_iij * d_iij3 / 4
					);

			// d^2E / dy_i^2
			gsl_matrix_set(m_dyidyi, i, j,
					(ry_i - ry_j) * ty_j * d_ij3 / 2
					+ 3 * gsl_pow_2(ry_i - ry_j) * tdt_ij * d_ij5 / 16
					- tdt_ij * d_ij3 / 4

					- (ry_ii - ry_j) * ty_j * d_iij3 / 2
					+ 3 * gsl_pow_2(ry_ii - ry_j) * tdt_iij * d_iij5 / 16
					- tdt_iij * d_iij3 / 4
					);

			// d^2E / dx_i dx_(i-1)
			gsl_matrix_set(m_dxidxii, i, j,
					3 * gsl_pow_2(rx_ii - rx_j) * tdt_iij * d_iij5 / 16
					- tdt_iij * d_iij3 / 4
					);

			// d^2E / dy_i dy_(i-1)
			gsl_matrix_set(m_dyidyii, i, j,
					3 * gsl_pow_2(ry_ii - ry_j) * tdt_iij * d_iij5 / 16
					- tdt_iij * d_iij3 / 4
					);

			gsl_matrix_set(m_dxidyj, i, j,
					(rx_i - rx_j) * ty_i * d_ij3 / 4
					- (ry_i - ry_j) * tx_j * d_ij3 / 4
					- 3 * (rx_i - rx_j) * (ry_i - ry_j) * tdt_ij * d_ij5 / 16

					+ (rx_ii - rx_j) * ty_ii * d_iij3 / 4
					+ (ry_ii - ry_j) * tx_j * d_iij3 / 4
					- 3 * (rx_ii - rx_j) * (ry_ii - ry_j) * tdt_iij * d_iij5 / 16

					- (rx_i - rx_jj) * ty_i * d_ijj3 / 4
					- (ry_i - ry_jj) * tx_jj * d_ijj3 / 4
					- 3 * (rx_i - rx_jj) * (ry_i - ry_jj) * tdt_ijj * d_ijj5 / 16

					- (rx_ii - rx_jj) * ty_ii * d_iijj3 / 4
					+ (ry_ii - ry_jj) * tx_jj * d_iijj3 / 4
					- 3 * (rx_ii - rx_jj) * (ry_ii - ry_jj) * tdt_iijj * d_iijj5 / 16
					);

			gsl_matrix_set(m_dxidyi, i, j,
					(ry_i - ry_j) * tx_j * d_ij3 / 4
					+ (rx_i - rx_j) * ty_j * d_ij3 / 4
					+ 3 * (rx_i - rx_j) * (ry_i - ry_j) * tdt_ij * d_ij5 / 16

					- (ry_ii - ry_j) * tx_j * d_iij3 / 4
					- (rx_ii - rx_j) * ty_j * d_iij3 / 4
					+ 3 * (rx_ii - rx_j) * (ry_ii - ry_j) * tdt_iij * d_iij5 / 16
					);

			gsl_matrix_set(m_dxiidyi, i, j,
					(ry_ii - ry_j) * tx_j * d_iij3 / 4
					- (rx_ii - rx_j) * ty_j * d_iij3 / 4
					+ 3 * (rx_ii - rx_j) * (ry_ii - ry_j) * tdt_iij * d_iij5 / 16
					);

			gsl_matrix_set(m_dxidyii, i, j,
					- (ry_i - ry_j) * tx_j * d_ij3 / 4
					+ (rx_i - rx_j) * ty_j * d_ij3 / 4
					+ 3 * (rx_i - rx_j) * (ry_i - ry_j) * tdt_ij * d_ij5 / 16
					);

		}
	}

		// Setting terms of d^2E / dy_i dy_j  and d^2E / dx_i dx_j in the Hessian.
	#pragma omp parallel for
	for (unsigned i = 2; i < n; i++) {
		for (unsigned j = 0; j < i - 1; j++) {

			gsl_matrix_set(hess, i, j, - gsl_matrix_get(m_dxidxj, i, j));

			gsl_matrix_set(hess, n + i, n + j, - gsl_matrix_get(m_dyidyj, i, j));
		}
	}

	// Zeroing out terms which aren't supposed to appear in the sums.
	#pragma omp parallel for private(ii, iii, jj)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);
		iii = gsl_permutation_get(indices2Right, i);
		jj = gsl_permutation_get(indicesLeft, i);
		gsl_matrix_set(m_dxidxi, i, i, 0);
		gsl_matrix_set(m_dxidxi, i, ii, 0);
		gsl_matrix_set(m_dyidyi, i, i, 0);
		gsl_matrix_set(m_dyidyi, i, ii, 0);
		gsl_matrix_set(m_dxidxii, i, i, 0);
		gsl_matrix_set(m_dxidxii, i, ii, 0);
		gsl_matrix_set(m_dxidxii, i, iii, 0);
		gsl_matrix_set(m_dyidyii, i, i, 0);
		gsl_matrix_set(m_dyidyii, i, ii, 0);
		gsl_matrix_set(m_dyidyii, i, iii, 0);
		gsl_matrix_set(m_dxidyi, i, i, 0);
		gsl_matrix_set(m_dxidyi, i, ii, 0);
		gsl_matrix_set(m_dxiidyi, i, i, 0);
		gsl_matrix_set(m_dxiidyi, i, ii, 0);
		gsl_matrix_set(m_dxiidyi, i, iii, 0);
		gsl_matrix_set(m_dxidyii, i, ii, 0);
		gsl_matrix_set(m_dxidyii, i, jj, 0);
		gsl_matrix_set(m_dxidyii, i, i, 0);
	}

	gsl_blas_dgemv(CblasNoTrans, 1, m_dxidxi, v_ones, 0, v_storage);

	// Setting terms of d^2E / dx_i^2 in the Hessian.
	#pragma omp parallel for private(ii, d_iij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		d_iij = gsl_matrix_get(dists, i, ii);

		gsl_matrix_set(hess, i, i,
				- (gsl_vector_get(v_storage, i) - 2 * d_iij)
				+ 2 * (ld[i] + ld[ii])
				);
	}

	gsl_blas_dgemv(CblasNoTrans, 1, m_dyidyi, v_ones, 0, v_storage);

	// Setting terms of d^2E / dy_i^2 in the Hessian.
	#pragma omp parallel for private(ii, d_iij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		d_iij = gsl_matrix_get(dists, i, ii);

		gsl_matrix_set(hess, n + i, n + i,
				- (gsl_vector_get(v_storage, i) - 2 * d_iij)
				+ 2 * (ld[i] + ld[ii])
				);
	}

	gsl_blas_dgemv(CblasNoTrans, 1, m_dxidxii, v_ones, 0, v_storage);

	#pragma omp parallel for private(ii, rx_i, rx_ii, tx_i, tx_ii, d_ij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		rx_i = gsl_vector_get(rx, i);
		rx_ii = gsl_vector_get(rx, ii);
		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);
		d_ij = gsl_matrix_get(dists, i, ii);
		gsl_vector_set(v_storage, i, gsl_vector_get(v_storage, i)
				- (rx_ii - rx_i) * (tx_i - tx_ii) * gsl_pow_3(d_ij) / 4
				+ d_ij);
	}

	#pragma omp parallel for private(ii, iii, rx_i, rx_ii, rx_iii, tx_i, tx_ii, tx_iii, d_ij, d_iij, d_iijj, tdt_ij)
	for (unsigned i = 0; i < n; i++) {
		iii = gsl_permutation_get(indices2Right, i);
		ii = gsl_permutation_get(indicesRight, i);

		rx_i = gsl_vector_get(rx, i);
		rx_ii = gsl_vector_get(rx, ii);
		rx_iii = gsl_vector_get(rx, iii);
		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);
		tx_iii = gsl_vector_get(tx, iii);
		d_ij = gsl_matrix_get(dists, i, ii);
		d_iij = gsl_matrix_get(dists, i, iii);
		d_iijj = gsl_matrix_get(dists, ii, iii);
		tdt_ij = gsl_matrix_get(tdots, i, iii);

		gsl_vector_set(v_storage, i, gsl_vector_get(v_storage, i)
				- 3 * gsl_pow_2(rx_iii - rx_i) * tdt_ij * gsl_pow_5(d_iij) / 16
				+ (tx_iii + tx_i) * (rx_iii - rx_i) * gsl_pow_3(d_iij) / 4
				+ tdt_ij * gsl_pow_3(d_iij) / 4 - d_iij
				- (tx_ii - tx_iii) * (rx_ii - rx_iii) * gsl_pow_3(d_iijj) / 4 + d_iijj);
	}

	gsl_matrix_set(hess, n - 1, 0,
			- gsl_vector_get(v_storage, 0) - 2 * ld[n - 1]);

	#pragma omp parallel for
	for (unsigned i = 1; i < n; i++) {
		gsl_matrix_set(hess, i,  i - 1,
				- gsl_vector_get(v_storage, i) - 2 * ld[i - 1]
				);
	}


	gsl_blas_dgemv(CblasNoTrans, 1, m_dyidyii, v_ones, 0, v_storage);

	#pragma omp parallel for private(ii, ry_i, ry_ii, ty_i, ty_ii, d_ij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		ry_i = gsl_vector_get(ry, i);
		ry_ii = gsl_vector_get(ry, ii);
		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);
		d_ij = gsl_matrix_get(dists, i, ii);
		gsl_vector_set(v_storage, i, gsl_vector_get(v_storage, i)
				- (ry_ii - ry_i) * (ty_i - ty_ii) * gsl_pow_3(d_ij) / 4
				+ d_ij);
	}

	#pragma omp parallel for private(ii, iii, ry_i, ry_ii, ry_iii, ty_i, ty_ii, ty_iii, d_ij, d_iij, d_iijj, tdt_ij)
	for (unsigned i = 0; i < n; i++) {
		iii = gsl_permutation_get(indices2Right, i);
		ii = gsl_permutation_get(indicesRight, i);

		ry_i = gsl_vector_get(ry, i);
		ry_ii = gsl_vector_get(ry, ii);
		ry_iii = gsl_vector_get(ry, iii);
		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);
		ty_iii = gsl_vector_get(ty, iii);
		d_ij = gsl_matrix_get(dists, i, ii);
		d_iij = gsl_matrix_get(dists, i, iii);
		d_iijj = gsl_matrix_get(dists, ii, iii);
		tdt_ij = gsl_matrix_get(tdots, i, iii);

		gsl_vector_set(v_storage, i, gsl_vector_get(v_storage, i)
				- 3 * gsl_pow_2(ry_iii - ry_i) * tdt_ij * gsl_pow_5(d_iij) / 16
				+ (ty_iii + ty_i) * (ry_iii - ry_i) * gsl_pow_3(d_iij) / 4
				+ tdt_ij * gsl_pow_3(d_iij) / 4 - d_iij
				- (ty_ii - ty_iii) * (ry_ii - ry_iii) * gsl_pow_3(d_iijj) / 4 + d_iijj);
	}

	#pragma omp parallel for
	for (unsigned i = 1; i < n; i++) {
		gsl_matrix_set(hess, n + i,  n + i - 1,
				- gsl_vector_get(v_storage, i) - 2 * ld[i - 1]
				);
	}

	gsl_matrix_set(hess, 2 * n -1, n,
			- gsl_vector_get(v_storage, 0) - 2 * ld[n-1]);


	// dxdy boring style
	#pragma omp parallel for
	for (unsigned j = 2; j < n; j++) {
		for (unsigned i = 0; i < j - 1; i++) {
			gsl_matrix_set(hess, n + j, i,
					- gsl_matrix_get(m_dxidyj, i, j));
		}
	}

	#pragma omp parallel for
	for (unsigned j = 0; j < n - 2; j++) {
		for (unsigned i = j + 2; i < n; i++) {
			gsl_matrix_set(hess, n + j, i,
					- gsl_matrix_get(m_dxidyj, i, j));
		}
	}

	// d^2E / dx_i dy_i

	gsl_blas_dgemv(CblasNoTrans, 1, m_dxidyi, v_ones, 0, v_storage);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		gsl_matrix_set(hess, n + i, i,
				- gsl_vector_get(v_storage, i)
				);
	}

	// d^2 E / dx_ii dy_i

	gsl_blas_dgemv(CblasNoTrans, 1, m_dxiidyi, v_ones, 0, v_storage);

	#pragma omp parallel for private(ii, iii, rx_i, rx_ii, rx_iii, tx_i, tx_ii, tx_iii, ry_i, ry_ii, ry_iii, ty_i, ty_ii, ty_iii, d_ij, d_iij, d_iijj, tdt_iij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);
		iii = gsl_permutation_get(indices2Right, i);

		rx_i = gsl_vector_get(rx, i);
		rx_ii = gsl_vector_get(rx, ii);
		rx_iii = gsl_vector_get(rx, iii);
		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);
		tx_iii = gsl_vector_get(tx, iii);
		ry_i = gsl_vector_get(ry, i);
		ry_ii = gsl_vector_get(ry, ii);
		ry_iii = gsl_vector_get(ry, iii);
		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);
		ty_iii = gsl_vector_get(ty, iii);

		d_ij = gsl_matrix_get(dists, i, ii);
		d_iij = gsl_matrix_get(dists, i, iii);
		d_iijj = gsl_matrix_get(dists, ii, iii);
		tdt_iij = gsl_matrix_get(tdots, i, iii);

		gsl_vector_set(v_storage, i, gsl_vector_get(v_storage, i)
				- (rx_ii - rx_i) * (ty_i - ty_ii) * gsl_pow_3(d_ij) / 4
				- 3 * (rx_iii - rx_i) * (ry_iii - ry_i) * tdt_iij * gsl_pow_5(d_iij) / 16
				+ (rx_iii - rx_i) * ty_iii * gsl_pow_3(d_iij) / 4
				+ (ry_iii - ry_i) * tx_i * gsl_pow_3(d_iij) / 4
				- (tx_ii - tx_iii) * (ry_ii - ry_iii) * gsl_pow_3(d_iijj) / 4
				);
	}

	#pragma omp parallel for
	for (unsigned i = 0; i < n - 1; i++) {
		gsl_matrix_set(hess, n + i + 1, i,
				- gsl_vector_get(v_storage, i + 1) - la / 2
				);
	}

	gsl_matrix_set(hess, n, n - 1,
			-gsl_vector_get(v_storage, 0) - la / 2);

	// Upper off-diagonal of dxdy submatrix.

	gsl_blas_dgemv(CblasNoTrans, 1, m_dxidyii, v_ones, 0, v_storage);

	#pragma omp parallel for private(ii, jj, rx_i, rx_ii, rx_jj, tx_i, tx_ii, tx_jj, ry_i, ry_ii, ry_jj, ty_i, ty_ii, ty_jj, d_ij, d_ijj, d_iijj, tdt_iijj)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);
		jj = gsl_permutation_get(indicesLeft, i);

		rx_i = gsl_vector_get(rx, i);
		rx_ii = gsl_vector_get(rx, ii);
		rx_jj = gsl_vector_get(rx, jj);
		tx_i = gsl_vector_get(tx, i);
		tx_ii = gsl_vector_get(tx, ii);
		tx_jj = gsl_vector_get(tx, jj);
		ry_i = gsl_vector_get(ry, i);
		ry_ii = gsl_vector_get(ry, ii);
		ry_jj = gsl_vector_get(ry, jj);
		ty_i = gsl_vector_get(ty, i);
		ty_ii = gsl_vector_get(ty, ii);
		ty_jj = gsl_vector_get(ty, jj);

		d_ij = gsl_matrix_get(dists, i, ii);
		d_ijj = gsl_matrix_get(dists, i, jj);
		d_iijj = gsl_matrix_get(dists, ii, jj);
		tdt_iijj = gsl_matrix_get(tdots, ii, jj);

		gsl_vector_set(v_storage, i, gsl_vector_get(v_storage, i)
				- (rx_i - rx_ii) * (ty_i - ty_ii) * gsl_pow_3(d_ij) / 4
				- (tx_jj - tx_i) * (ry_i - ry_jj) * gsl_pow_3(d_ijj) / 4
				+ tx_ii * (ry_ii - ry_jj) * gsl_pow_3(d_iijj) / 4
				- (rx_jj - rx_ii) * ty_jj * gsl_pow_3(d_iijj) / 4
				+ 3 * (rx_jj - rx_ii) * (ry_ii - ry_jj) * tdt_iijj * gsl_pow_5(d_iijj) / 16
				);
	}

	gsl_matrix_set(hess, 2 * n - 1, 0,
			- gsl_vector_get(v_storage, n - 1) + la / 2
			);


	#pragma omp parallel for
	for (unsigned i = 0; i < n-1; i++) {
		gsl_matrix_set(hess, n + i, i + 1,
				-  gsl_vector_get(v_storage, i) + la / 2
				);
	}


	// dLdL

	double gradLDist = 0;
	for (unsigned i = 0; i < n; i++) {
		gradLDist += 2 * ld[i] / gsl_pow_2(n);
	}

	L = fabs(L);

	double gradL = - 1 / L - gradLDist;

	gsl_matrix_set(hess, 2 * n, 2 * n, gradL);

	// dLdlambdad

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		gsl_matrix_set(hess, i + 2 * n + 2, 2 * n,  - 2 * L / gsl_pow_2(n));
	}

	#pragma omp parallel for private(ii)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		gsl_matrix_set(hess, 2 * n + 1, i, -
				0.5 * (gsl_vector_get(ty, i) + gsl_vector_get(ty, ii)));

		gsl_matrix_set(hess, 2 * n + 1,  n + i, 
				0.5 * (gsl_vector_get(tx, i) + gsl_vector_get(tx, ii)));
	}

	#pragma omp parallel for private(ii)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		gsl_matrix_set(hess, 2 * n + 2 + i, i, -
				2 * (gsl_vector_get(tx, i)));

		gsl_matrix_set(hess, 2 * n + 2 + i, n + i, -
				2 * (gsl_vector_get(ty, i)));
	}

	#pragma omp parallel for private(ii)
	for (unsigned i = 1; i < n; i++) {
		ii = gsl_permutation_get(indicesRight, i);

		gsl_matrix_set(hess, 2 * n + i + 1, i,
				2 * gsl_vector_get(tx, ii));

		gsl_matrix_set(hess, 2 * n + i + 1, n + i,
				2 * gsl_vector_get(ty, ii));
	}

	gsl_matrix_set(hess, 3 * n + 1, 0,
			2 * gsl_vector_get(tx, n-1));
	gsl_matrix_set(hess, 3 * n + 1, n,
			2 * gsl_vector_get(ty, n-1));


	gsl_vector_free(v_ones);
	gsl_vector_free(v_storage);

	gsl_matrix_free(m_dxidxj);
	gsl_matrix_free(m_dyidyj);
	gsl_matrix_free(m_dxidxi);
	gsl_matrix_free(m_dyidyi);
	gsl_matrix_free(m_dxidxii);
	gsl_matrix_free(m_dyidyii);
	gsl_matrix_free(m_dxidyj);
	gsl_matrix_free(m_dxidyi);
	gsl_matrix_free(m_dxiidyi);
	gsl_matrix_free(m_dxidyii);

	gsl_permutation_free(indicesLeft);
	gsl_permutation_free(indicesRight);
	gsl_permutation_free(indices2Right);
}


double domain_energy_nakedEnergy(unsigned n, const gsl_vector *z, double c) {
	double lagrangian;

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));
	double L = gsl_vector_get(z, 2 * n);

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	lagrangian = domain_energy_energy(n, c, rx, ry, tx, ty, tdots, dists, L);

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);

	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);

	return lagrangian;
};


double domain_energy_nakedLagrangian(unsigned n, const gsl_vector *z, double c) {
	double lagrangian;

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));
	double L = gsl_vector_get(z, 2 * n);
	double la = gsl_vector_get(z, 2 * n + 1);
	double ld[n];
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		ld[i] = gsl_vector_get(z, 2 * n + 2 + i);
	}

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	double area = domain_energy_area(n, x, y);

	lagrangian = domain_energy_lagrangian(n, c, rx, ry, tx, ty, tdots, dists, L, area, la, ld);

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);

	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);

	return lagrangian;
};


void domain_energy_nakedGradient(gsl_vector *grad, unsigned n, const gsl_vector *z, double c) {

	gsl_permutation *indices_right;
	indices_right = gsl_permutation_alloc(n);

	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	// Setting pointers to give the elements of z more convenient names.
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));
	double L = gsl_vector_get(z, 2 * n);
	double la = gsl_vector_get(z, 2 * n + 1);
	double ld[n];
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		ld[i] = gsl_vector_get(z, 2 * n + 2 + i);
	}

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	double area = domain_energy_area(n, x, y);

	domain_energy_gradient(grad, n, c, rx, ry, tx, ty, tdots, dists, L, area, la, ld);

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);
	gsl_permutation_free(indices_right);

}


void domain_energy_nakedHalfHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c) {

	gsl_permutation *indices_right;
	indices_right = gsl_permutation_alloc(n);

	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	// Setting pointers to give the elements of z more convenient names.
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));
	double L = gsl_vector_get(z, 2 * n);
	double la = gsl_vector_get(z, 2 * n + 1);
	double ld[n];
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		ld[i] = gsl_vector_get(z, 2 * n + 2 + i);
	}

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	double area = domain_energy_area(n, x, y);

	domain_energy_halfHessian(hess, n, c, rx, ry, tx, ty, tdots, dists, L, la, ld);

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);
	gsl_permutation_free(indices_right);
}


void domain_energy_nakedHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c) {

	domain_energy_nakedHalfHessian(hess, n, z, c);

	#pragma omp parallel for
	for (unsigned i = 1; i < 3 * n + 2; i++) {
		for (unsigned j = 0; j < i; j++) {
			gsl_matrix_set(hess, j, i, gsl_matrix_get(hess, i, j));
		}
	}
}


double domain_energy_fixedEnergy(unsigned n, const gsl_vector *z, double c) {
	double lagrangian;

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	gsl_vector_set(y, 0, 0);
	#pragma omp parallel for
	for (unsigned i = 0; i < n - 1; i++) gsl_vector_set(y, i + 1, gsl_vector_get(z, i + n));
	double L = gsl_vector_get(z, 2 * n - 1);

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	lagrangian = domain_energy_energy(n, c, rx, ry, tx, ty, tdots, dists, L);

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);

	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);

	return lagrangian;
};


// The fixed functions.

double domain_energy_fixedLagrangian(unsigned n, const gsl_vector *z, double c) {
	double lagrangian;

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	gsl_vector_set(y, 0, 0);
	#pragma omp parallel for
	for (unsigned i = 0; i < n - 1; i++) gsl_vector_set(y, i + 1, gsl_vector_get(z, i + n));
	double L = gsl_vector_get(z, 2 * n - 1);
	double la = gsl_vector_get(z, 2 * n);
	double ld[n];
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		ld[i] = gsl_vector_get(z, 2 * n + 1 + i);
	}
	double lx = gsl_vector_get(z, 3 * n + 1);
	double ly = gsl_vector_get(z, 3 * n + 2);

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	domain_energy_rt(rx, n, x, 1);
	domain_energy_rt(ry, n, y, 1);
	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	domain_energy_tdots(tdots, n, tx, ty);
	domain_energy_dists(dists, n, rx, ry);

	double area = domain_energy_area(n, x, y);

	lagrangian = domain_energy_lagrangian(n, c, rx, ry, tx, ty, tdots, dists, L, area, la, ld);

	double xtot = 0;
	for (unsigned i = 0; i < n; i++) xtot += gsl_vector_get(z, i);

	double ytot = 0;
	for (unsigned i = 1; i < n; i++) ytot += gsl_vector_get(z, i + n - 1);

	lagrangian += - lx * xtot - ly * ytot;

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);

	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);

	return lagrangian;
};


void domain_energy_fixedGradient(gsl_vector *grad, unsigned n, const gsl_vector *z,
		double c) {

	// Setting pointers to give the elements of z more convenient names.
	double L = gsl_vector_get(z, 2 * n - 1);
	double la = gsl_vector_get(z, 2 * n);
	double ld[n];
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		ld[i] = gsl_vector_get(z, 2 * n + 1 + i);
	}
	double lx = gsl_vector_get(z, 3 * n + 1);
	double ly = gsl_vector_get(z, 3 * n + 2);

	unsigned ii;

	gsl_vector *rx, *ry, *tx, *ty, *freegrad;
	gsl_matrix *tdots, *dists;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	freegrad = gsl_vector_alloc(3 * n + 2);

	double area = domain_energy_init(n, z, rx, ry, tx, ty, tdots, dists);

	domain_energy_gradient(freegrad, n, c, rx, ry, tx, ty, tdots, dists, L, area, la, ld);

	#pragma omp parallel for private(ii)
	for (unsigned i =0; i < 3 * n + 2; i++) {
		if (i != n) {
			if (i < n) ii = i;
			if (i > n) ii = i - 1;

			gsl_vector_set(grad, ii, gsl_vector_get(freegrad, i));
		}
	}


	#pragma omp parallel for private(ii)
	for (unsigned i = 0; i < n; i++) {
		gsl_vector_set(grad, i, gsl_vector_get(grad, i) - lx);
		if (i != 0) gsl_vector_set(grad, n + i - 1, gsl_vector_get(grad, i + n - 1) -ly);
	}


	double xtot = 0;
	for (unsigned i = 0; i < n; i++) xtot += gsl_vector_get(z, i);
	double ytot = 0;
	for (unsigned i = 1; i < n; i++) ytot += gsl_vector_get(z, i + n - 1);

	gsl_vector_set(grad, 3 * n + 1, -xtot);

	gsl_vector_set(grad, 3 * n + 2, -ytot);

	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_vector_free(freegrad);
	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);

}


void domain_energy_fixedHalfHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z,
		double c) {

	// Setting pointers to give the elements of z more convenient names.
	double L = gsl_vector_get(z, 2 * n - 1);
	double la = gsl_vector_get(z, 2 * n);
	double ld[n];
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		ld[i] = gsl_vector_get(z, 2 * n + 1 + i);
	}
	double lx = gsl_vector_get(z, 3 * n + 1);
	double ly = gsl_vector_get(z, 3 * n + 2);

	gsl_vector *rx, *ry, *tx, *ty;
	gsl_matrix *tdots, *dists, *freehess;

	rx = gsl_vector_alloc(n);
	ry = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	tdots = gsl_matrix_alloc(n, n);
	dists = gsl_matrix_alloc(n, n);

	freehess = gsl_matrix_alloc(3 * n + 2, 3 * n + 2);

	double area = domain_energy_init(n, z, rx, ry, tx, ty, tdots, dists);

	domain_energy_halfHessian(freehess, n, c, rx, ry, tx, ty, tdots, dists, L, la, ld);

	gsl_matrix *m_dxidxj, *m_dyidyj, *m_dxidxi, *m_dyidyi, *m_dxidxii,
						 *m_dyidyii, *m_dxidyj, *m_dxidyi, *m_dxiidyi, *m_dxidyii;
	gsl_vector *v_ones, *v_storage;
	gsl_permutation *indicesRight, *indicesLeft, *indices2Right;

	unsigned ii, jj, iii;
	double rx_i, rx_j, tx_i, tx_j, tdt_ij, d_ij, rx_ii, rx_jj, tx_ii, tx_jj,
				 ry_i, ry_j, ty_i, ty_j, ry_ii, ry_jj, ty_ii, ty_jj, tdt_iij, d_iij,
				 tdt_ijj, d_ijj, tdt_iijj, d_iijj, rx_iii, tx_iii, ry_iii, ty_iii,
				 d_ij3, d_iij3, d_ijj3, d_iijj3, d_ij5, d_iij5, d_ijj5, d_iijj5;

	gsl_matrix_set_zero(hess);

	#pragma omp parallel for private(ii, jj)
	for (unsigned i = 0; i < 3 * n + 2; i++) {
		if (i != n) {
			if (i < n) ii = i;
			if (i > n) ii = i - 1;
			for (unsigned j = 0; j <= i; j++) {
				if (j != n) {
					if (j < n) jj = j;
					if (j > n) jj = j - 1;

					gsl_matrix_set(hess, ii, jj, gsl_matrix_get(freehess, i, j));
				}
			}
		}
	}

	indicesRight = gsl_permutation_alloc(n);
	indicesLeft = gsl_permutation_alloc(n);
	indices2Right = gsl_permutation_alloc(n);

	gsl_permutation_init(indicesRight);
	gsl_permutation_init(indicesLeft);
	gsl_permutation_over(n, indicesRight, true);
	gsl_permutation_over(n, indicesLeft, false);
	gsl_permutation_memcpy(indices2Right, indicesRight);
	gsl_permutation_over(n, indices2Right, true);


	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) {
		gsl_matrix_set(hess, 3 * n + 1, i, -1);
		if (i != 0) gsl_matrix_set(hess, 3 * n + 2, n + i - 1, -1);
	}

	gsl_vector_free(rx);
	gsl_vector_free(ry);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_matrix_free(tdots);
	gsl_matrix_free(dists);
	gsl_matrix_free(freehess);

	gsl_permutation_free(indicesLeft);
	gsl_permutation_free(indicesRight);
	gsl_permutation_free(indices2Right);
}


void domain_energy_fixedHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z,
		double c) {

	domain_energy_fixedHalfHessian(hess, n, z, c);

	#pragma omp parallel for
	for (unsigned i = 1; i < 3 * n + 3; i++) {
		for (unsigned j = 0; j < i; j++) {
			gsl_matrix_set(hess, j, i, gsl_matrix_get(hess, i, j));
		}
	}
}


// The random functions.

double domain_energy_randEnergy(unsigned n, const gsl_vector *z,
	unsigned ord, const gsl_vector *k, const gsl_vector *a,
	const gsl_vector *phi) {

	double energy;

	gsl_vector *x, *y;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));

	unsigned ii;

	gsl_vector *tx, *ty;

	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	energy = 0;

	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < ord; j++) {
			energy += gsl_vector_get(a, j) * gsl_sf_sin(gsl_vector_get(k, j) * gsl_vector_get(x, i) + gsl_vector_get(k, j + ord) * gsl_vector_get(y, i) + gsl_vector_get(phi, j)) * (gsl_vector_get(ty, i) / gsl_vector_get(k, j) - gsl_vector_get(tx, i) / gsl_vector_get(k, j + ord)) / 2;
		}
	}
	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(tx);
	gsl_vector_free(ty);

	return energy;
};


void domain_energy_randGradient(gsl_vector *grad, unsigned n,
	const gsl_vector *z, unsigned ord, const gsl_vector *k,
	const gsl_vector *a, const gsl_vector *phi) {

	gsl_permutation *indices_right;
	indices_right = gsl_permutation_alloc(n);

	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);

	gsl_vector *x, *y, *tx, *ty;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));

	unsigned ii;

	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	double thesumx, thesumy, aj, kxj, kyj, phij, xi, yi, xii, yii, txi, tyi;

	#pragma omp parallel for private(ii, xi, yi, xii, yii, txi, tyi, thesumx, thesumy, aj, kxj, kyj, phij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);

		xi = gsl_vector_get(x, i);
		yi = gsl_vector_get(y, i);
		xii = gsl_vector_get(x, ii);
		yii = gsl_vector_get(y, ii);
		txi = gsl_vector_get(tx, i);
		tyi = gsl_vector_get(ty, i);

		thesumx = 0;
		thesumy = 0;

		for (unsigned j = 0; j < ord; j++) {
			aj = gsl_vector_get(a, j);
			kxj = gsl_vector_get(k, j);
			kyj = gsl_vector_get(k, ord + j);
			phij = gsl_vector_get(phi, j);

			thesumx += aj * (kxj * gsl_sf_cos(kxj * xi + kyj * yi + phij) * (tyi / kxj - txi / kyj) +
					(gsl_sf_sin(kxj * xi + kyj * yi + phij) - gsl_sf_sin(kxj * xii + kyj * yii + phij)) / kyj);

			thesumy += aj * (kyj * gsl_sf_cos(kxj * xi + kyj * yi + phij) * (tyi / kxj - txi / kyj) -
					(gsl_sf_sin(kxj * xi + kyj * yi + phij) - gsl_sf_sin(kxj * xii + kyj * yii + phij)) / kxj);
		}

		gsl_vector_set(grad, i, gsl_vector_get(grad, i) + thesumx / 2);
		gsl_vector_set(grad, n + i, gsl_vector_get(grad, n + i) + thesumy / 2);
	}

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_permutation_free(indices_right);
}


void domain_energy_randHalfHessian(gsl_matrix *hess, unsigned n,
	const gsl_vector *z, unsigned ord, const gsl_vector *k, const gsl_vector *a,
	const gsl_vector *phi) {

	gsl_permutation *indices_right;
	indices_right = gsl_permutation_alloc(n);

	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);

	gsl_vector *x, *y, *tx, *ty;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));

	unsigned ii;

	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	double thesumx, thesumy, thesumxy, thesumxx, thesumyy, thesumxxy, thesumxyy, aj, kxj, kyj, phij, xi, yi, xii, yii, txi, tyi;

	#pragma omp parallel for private(ii, xi, yi, xii, yii, txi, tyi, thesumx, thesumy, thesumxy, thesumxx, thesumyy, thesumxxy, thesumxyy, aj, kxj, kyj, phij)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);

		xi = gsl_vector_get(x, i);
		yi = gsl_vector_get(y, i);
		xii = gsl_vector_get(x, ii);
		yii = gsl_vector_get(y, ii);
		txi = gsl_vector_get(tx, i);
		tyi = gsl_vector_get(ty, i);

		thesumx = 0;
		thesumy = 0;
		thesumxy = 0;
		thesumxx = 0;
		thesumyy = 0;
		thesumxxy = 0;
		thesumxyy = 0;

		for (unsigned j = 0; j < ord; j++) {
			aj = gsl_vector_get(a, j);
			kxj = gsl_vector_get(k, j);
			kyj = gsl_vector_get(k, ord + j);
			phij = gsl_vector_get(phi, j);

			thesumx += aj * ( - gsl_pow_2(kxj) * gsl_sf_sin(kxj * xi + kyj * yi + phij) * (tyi / kxj - txi / kyj) +
					2 * gsl_sf_cos(kxj * xi + kyj * yi + phij) * kxj / kyj);

			thesumy += aj * ( - gsl_pow_2(kyj) * gsl_sf_sin(kxj * xi + kyj * yi + phij) * (tyi / kxj - txi / kyj) -
					2 * gsl_sf_cos(kxj * xi + kyj * yi + phij) * kyj / kxj);

			thesumxy += - aj * kxj * kyj * gsl_sf_sin(kxj * xi + kyj * yi + phij) * (tyi / kxj - txi / kyj);

			thesumxx += - aj * gsl_sf_cos(kxj * xii + kyj * yii + phij) * kxj / kyj;

			thesumyy += aj * gsl_sf_cos(kxj * xii + kyj * yii + phij) * kyj / kxj;

			thesumxyy += - aj * gsl_sf_cos(kxj * xii + kyj * yii + phij);

			thesumxxy += aj * gsl_sf_cos(kxj * xii + kyj * yii + phij);
		}

		gsl_matrix_set(hess, i, i, gsl_matrix_get(hess, i, i) + thesumx / 2);
		gsl_matrix_set(hess, n + i, n + i, gsl_matrix_get(hess, n + i, n + i) + thesumy / 2);
		gsl_matrix_set(hess, i + n, i, gsl_matrix_get(hess, n + i, i) + thesumxy / 2);
		if (i == 0) {
			gsl_matrix_set(hess, n - 1, 0, gsl_matrix_get(hess, n - 1, 0) + thesumxx / 2);
			gsl_matrix_set(hess, 2 * n - 1, n, gsl_matrix_get(hess, 2 * n - 1, n) + thesumyy / 2);
		} else {
			gsl_matrix_set(hess, i, ii, gsl_matrix_get(hess, i, ii) + thesumxx / 2);
			gsl_matrix_set(hess, n + i, n + ii, gsl_matrix_get(hess, n + i, n + ii) + thesumyy / 2);
		}
		gsl_matrix_set(hess, n + i, ii, gsl_matrix_get(hess, n + i, ii) + thesumxxy / 2);
		gsl_matrix_set(hess, n + ii, i, gsl_matrix_get(hess, n + ii, i) + thesumxyy / 2);
	}
	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_permutation_free(indices_right);
}


// The random naked functions.

double domain_energy_nakedRandLagrangian(unsigned n, const gsl_vector *z,
	double c, unsigned ord,  const gsl_vector *k, const gsl_vector *a,
	const gsl_vector *phi) {

	double lagrangian, randEnergy;

	lagrangian = domain_energy_nakedLagrangian(n, z, c);
	randEnergy = domain_energy_randEnergy(n, z, ord, k, a, phi);

	return lagrangian + randEnergy;
}


void domain_energy_nakedRandGradient(gsl_vector *grad, unsigned n,
	const gsl_vector *z, double c, unsigned ord,  const gsl_vector *k,
	const gsl_vector *a, const gsl_vector *phi) {

	domain_energy_nakedGradient(grad, n, z, c);
	domain_energy_randGradient(grad, n, z, ord, k, a, phi);
}


void domain_energy_nakedRandHessian(gsl_matrix *hess, unsigned n,
	const gsl_vector *z, double c, unsigned ord,  const gsl_vector *k,
	const gsl_vector *a, const gsl_vector *phi) {

	domain_energy_nakedHalfHessian(hess, n, z, c);
	domain_energy_randHalfHessian(hess, n, z, ord, k, a, phi);

	#pragma omp parallel for
	for (unsigned i = 1; i < 3 * n + 2; i++) {
		for (unsigned j = 0; j < i; j++) {
			gsl_matrix_set(hess, j, i, gsl_matrix_get(hess, i, j));
		}
	}
}


// The well functions.

double domain_energy_wellEnergy(unsigned n, const gsl_vector *z, double w,
	double s) {

	double xi, yi, txi, tyi, energy;
	gsl_vector *x, *y, *tx, *ty;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i,
		gsl_vector_get(z, i + n));

	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	energy = 0;

	for (unsigned i = 0; i < n; i++) {
		xi = gsl_vector_get(x, i);
		yi = gsl_vector_get(y, i);
		txi = gsl_vector_get(tx, i);
		tyi = gsl_vector_get(ty, i);

		energy += ((gsl_sf_exp(s * (xi - w)) - gsl_sf_exp(-s * (xi + w))) * tyi -
			(gsl_sf_exp(s * (yi - w)) - gsl_sf_exp(-s * (yi + w))) * txi) / s;
	}

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(tx);
	gsl_vector_free(ty);

	return energy;
};


void domain_energy_wellGradient(gsl_vector *grad, unsigned n,
	const gsl_vector *z, double w, double s) {

	unsigned ii;
	double xi, yi, xii, yii, txi, tyi, grad_xi, grad_yi;
	gsl_vector *x, *y, *tx, *ty;
	gsl_permutation *indices_right;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);
	indices_right = gsl_permutation_alloc(n);

	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	#pragma omp parallel for
	for (unsigned i = 0; i < n; i++) gsl_vector_set(y, i, gsl_vector_get(z, i + n));

	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);

	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);


	#pragma omp parallel for private(ii, xi, yi, xii, yii, txi, tyi, grad_xi,\
		grad_yi)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);

		xi = gsl_vector_get(x, i);
		yi = gsl_vector_get(y, i);
		xii = gsl_vector_get(x, ii);
		yii = gsl_vector_get(y, ii);
		txi = gsl_vector_get(tx, i);
		tyi = gsl_vector_get(ty, i);

		grad_xi = tyi * (gsl_sf_exp(s * (xi - w)) + gsl_sf_exp(-s * (xi + w))) +
			(gsl_sf_exp(s * (yi - w)) - gsl_sf_exp(-s * (yi + w)) -
			 gsl_sf_exp(s * (yii - w)) + gsl_sf_exp(-s * (yii + w))) / s;

		grad_yi = - txi * (gsl_sf_exp(s * (yi - w)) + gsl_sf_exp(-s * (yi + w))) +
			(- gsl_sf_exp(s * (xi - w)) + gsl_sf_exp(-s * (xi + w)) +
			 gsl_sf_exp(s * (xii - w)) - gsl_sf_exp(-s * (xii + w))) / s;

		gsl_vector_set(grad, i, gsl_vector_get(grad, i) + grad_xi);

		gsl_vector_set(grad, n + i, gsl_vector_get(grad, n + i) + grad_yi);
	}

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_permutation_free(indices_right);
}


void domain_energy_wellHalfHessian(gsl_matrix *hess, unsigned n,
	const gsl_vector *z, double w, double s) {

	unsigned ii;
	double xi, yi, xii, yii, txi, tyi, hess_xi, hess_yi, hess_xiyi, hess_xiiyi,
		hess_xiyii, exp_mxi, exp_pxi, exp_myi, exp_pyi, exp_mxii, exp_myii,
		exp_pxii, exp_pyii;
	gsl_vector *x, *y, *tx, *ty;
	gsl_permutation *indices_right;

	x = gsl_vector_alloc(n);
	y = gsl_vector_alloc(n);
	tx = gsl_vector_alloc(n);
	ty = gsl_vector_alloc(n);
	indices_right = gsl_permutation_alloc(n);

	for (unsigned i = 0; i < n; i++) gsl_vector_set(x, i, gsl_vector_get(z, i));
	for (unsigned i = 0; i < n; i++) {
		gsl_vector_set(y, i, gsl_vector_get(z, i + n));
	}

	domain_energy_rt(tx, n, x, -1);
	domain_energy_rt(ty, n, y, -1);

	gsl_permutation_init(indices_right);
	gsl_permutation_over(n, indices_right, true);

	#pragma omp parallel for private(ii, xi, yi, xii, yii, txi, tyi, hess_xi,\
		hess_yi, hess_xiyi, hess_xiiyi, hess_xiyii, exp_mxi, exp_pxi, exp_myi,\
		exp_pyi, exp_mxii, exp_myii, exp_pxii, exp_pyii)
	for (unsigned i = 0; i < n; i++) {
		ii = gsl_permutation_get(indices_right, i);

		xi = gsl_vector_get(x, i);
		yi = gsl_vector_get(y, i);
		xii = gsl_vector_get(x, ii);
		yii = gsl_vector_get(y, ii);
		txi = gsl_vector_get(tx, i);
		tyi = gsl_vector_get(ty, i);
		exp_mxi = gsl_sf_exp(-s * (xi + w));
		exp_pxi = gsl_sf_exp(s * (xi - w));
		exp_myi = gsl_sf_exp(-s * (yi + w));
		exp_pyi = gsl_sf_exp(s * (yi - w));
		exp_mxii = gsl_sf_exp(-s * (xii + w));
		exp_pxii = gsl_sf_exp(s * (xii - w));
		exp_myii = gsl_sf_exp(-s * (yii + w));
		exp_pyii = gsl_sf_exp(s * (yii - w));

		hess_xi = tyi * s * (exp_pxi - exp_mxi);
		hess_yi = - txi * s * (exp_pyi - exp_myi);

		hess_xiyi = - exp_pxi - exp_mxi + exp_pyi + exp_myi;

		hess_xiyii = - exp_pyii - exp_myii;
		hess_xiiyi = exp_pxii + exp_mxii;

		gsl_matrix_set(hess, i, i, gsl_matrix_get(hess, i, i) + hess_xi);
		gsl_matrix_set(hess, n + i, n + i, gsl_matrix_get(hess, n + i, n + i) +
			hess_yi);
		gsl_matrix_set(hess, i + n, i, gsl_matrix_get(hess, n + i, i) + hess_xiyi);
		gsl_matrix_set(hess, n + i, ii, gsl_matrix_get(hess, n + i, ii) +
			hess_xiiyi);
		gsl_matrix_set(hess, n + ii, i, gsl_matrix_get(hess, n + ii, i) +
			hess_xiyii);
	}

	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_vector_free(tx);
	gsl_vector_free(ty);
	gsl_permutation_free(indices_right);
}


// The naked well functions.

double domain_energy_nakedWellLagrangian(unsigned n, const gsl_vector *z,
	double c, double w, double s) {

	double nakedLagrangian, wellEnergy;

	nakedLagrangian = domain_energy_nakedLagrangian(n, z, c);
	wellEnergy = domain_energy_wellEnergy(n, z, w, s);

	return nakedLagrangian + wellEnergy;
}


void domain_energy_nakedWellGradient(gsl_vector *grad, unsigned n,
	const gsl_vector *z, double c, double w, double s) {

	domain_energy_nakedGradient(grad, n, z, c);
	domain_energy_wellGradient(grad, n, z, w, s);
}


void domain_energy_nakedWellHessian(gsl_matrix *hess, unsigned n,
	const gsl_vector *z, double c, double w, double s) {

	domain_energy_nakedHalfHessian(hess, n, z, c);
	domain_energy_wellHalfHessian(hess, n, z, w, s);

	#pragma omp parallel for
	for (unsigned i = 1; i < 3 * n + 2; i++) {
		for (unsigned j = 0; j < i; j++) {
			gsl_matrix_set(hess, j, i, gsl_matrix_get(hess, i, j));
		}
	}
}


// The random well functions.

double domain_energy_randWellLagrangian(unsigned n, const gsl_vector *z,
	double c, unsigned ord,  const gsl_vector *k, const gsl_vector *a,
	const gsl_vector *phi, double w, double s) {

	double lagrangian, randEnergy, wellEnergy;

	lagrangian = domain_energy_nakedLagrangian(n, z, c);
	randEnergy = domain_energy_randEnergy(n, z, ord, k, a, phi);
	wellEnergy = domain_energy_wellEnergy(n, z, w, s);

	return lagrangian + randEnergy + wellEnergy;
}


void domain_energy_randWellGradient(gsl_vector *grad, unsigned n,
	const gsl_vector *z, double c, unsigned ord,  const gsl_vector *k,
	const gsl_vector *a, const gsl_vector *phi, double w, double s) {

	domain_energy_nakedGradient(grad, n, z, c);
	domain_energy_randGradient(grad, n, z, ord, k, a, phi);
	domain_energy_wellGradient(grad, n, z, w, s);
}


void domain_energy_randWellHessian(gsl_matrix *hess, unsigned n,
	const gsl_vector *z, double c, unsigned ord,  const gsl_vector *k,
	const gsl_vector *a, const gsl_vector *phi, double w, double s) {

	domain_energy_nakedHalfHessian(hess, n, z, c);
	domain_energy_randHalfHessian(hess, n, z, ord, k, a, phi);
	domain_energy_wellHalfHessian(hess, n, z, w, s);

	#pragma omp parallel for
	for (unsigned i = 1; i < 3 * n + 2; i++) {
		for (unsigned j = 0; j < i; j++) {
			gsl_matrix_set(hess, j, i, gsl_matrix_get(hess, i, j));
		}
	}
}
