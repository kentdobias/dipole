#ifndef DOMAIN_NEWTON_H
#define DOMAIN_NEWTON_H

#include <math.h>
#include <iostream>
#include <string>

// GSL includes.
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf.h>

// Eigen's linear solving uses cheap parallelization.
#include <eigen3/Eigen/Dense>

/* This function is templated so that any set of functions which return an
 * energy, gradient, and Hessian given an empty object, the size of the state
 * vector, and the state vector can be used.  This allows many such sets of
 * functions, e.g., that for a fixed domain or a domain on a random background,
 * to be used.  See the file domain_minimize.cpp for examples of construction
 * of these functions.
 */
template <class energy_func, class grad_func, class hess_func>

int domain_newton(gsl_vector *state, unsigned size, unsigned params,
		energy_func get_energy, grad_func get_grad, hess_func get_hess, double
		epsilon, unsigned max_iterations, double beta, double s, double sigma,
		double gamma, double eta_0, double delta, double bound, bool verbose, bool
		save_states) {
/* The function domain_newton carries out a modified version of Newton's
 * method.  On success, 0 is returned.  On failure, 1 is returned.
 * 
 * state          - GSL_VECTOR
 *                On entry, state gives the system's initial condition.  On
 *                exit, state contains the result Newton's method.
 *
 * size           - UNSIGNED INTEGER
 *                On entry, size gives the size of the vector state.  Unchanged
 *                on exit.
 *
 * params         - UNSIGNED INTEGER
 *                On entry, params gives the number of non-multiplier elements
 *                in state, which are assumed by the function to be the first
 *                elements of state.  Unchanged on exit.
 *
 * get_energy     - ENERGY_FUNC
 *                On entry, get_energy is a function that returns a double
 *                float.  The first argument of get_energy is an unsigned
 *                integer and the second argument is a gsl_vector object.  This
 *                function is expected to take size and state, respectively,
 *                and return the energy of that state.  Unchanged on exit.
 *
 * get_grad       - GRAD_FUNC
 *                On entry, get_grad is a function that returns void.  The
 *                first argument of get_grad is a gsl_vector object, the second
 *                argument of get_grad is an unsigned integer, and the third
 *                argument of get_grad is a gsl_vector object.  This function
 *                is expected to take a vector of size size, size, and state,
 *                respectively.  It leaves the gradient of the energy function
 *                in the first argument.  Unchanged on exit.
 *
 * get_hess       - HESS_FUNC
 *                On entry, get_hess is a function that returns void.  The
 *                first argument of get_hess is a gsl_matrix object, the second
 *                argument of get_hess is an unsigned integer, and the third
 *                argument of get_hess is a gsl_vector object.  This function
 *                is expected to take a matrix of size size by size, size, and
 *                state, respectively.  It leaves the Hessian of the energy
 *                function in the first argument.  Unchanged on exit.
 *
 * epsilon        - DOUBLE FLOAT
 *                On entry, epsilon gives the number that is used to judge
 *                convergence.  When the norm of the gradient is less than
 *                epsilon * size, the process is deemed complete and the
 *                iterations are stopped.  Unchanged on exit.
 *
 * max_iterations - UNSIGNED INTEGER
 *                On entry, max_iterations gives the maximum number of times
 *                the algorithm will repeat before failing.  Unchanged on exit.
 *
 * beta           - DOUBLE FLOAT
 *                On entry, beta gives the number which is exponentiated to
 *                scale the step size in Newton's method.  Unchanged on exit.
 *
 * s              - DOUBLE FLOAT
 *                On entry, s gives a constant scaling of the step size in
 *                Newton's method.  Unchanged on exit.
 *
 * sigma          - DOUBLE FLOAT
 *                On entry, sigma gives a scaling to the condition on the step
 *                size in Newton's method.  Unchanged on exit.
 *
 * gamma          - DOUBLE FLOAT
 *                On entry, gamma gives the amount by which the norm of the
 *                gradient must change for eta to decrement by a factor delta.
 *                Unchanged on exit.
 *
 * eta_0          - DOUBLE FLOAT
 *                On entry, eta_0 gives the starting value of eta.  Unchanged
 *                on exit.
 *
 * delta          - DOUBLE FLOAT
 *                On entry, delta gives the factor by which eta is decremented.
 *                Unchanged on exit.
 *
 * bound          - DOUBLE FLOAT
 *                On entry, delta gives an upper bound to the gradient norm.
 *                If surpassed, the execution is halted and the program returns
 *                failure.  Unchanged on exit.
 *
 * verbose        - BOOLEAN
 *                On entry, verbose indicates whether verbose output will be
 *                printed to stdout by this program.  Unchanged on exit.
 */

	// Declaring variables.
	double ratio, norm, old_norm, old_energy, energy, grad_dz_prod, alpha, eta;
	unsigned iterations, m;
	bool converged, bound_exceeded;

	// Declaring GSL variables.
	gsl_vector *grad, *dz;
	gsl_matrix *hess;

	// Allocating memory for GSL objects
	grad = gsl_vector_alloc(size);
	dz = gsl_vector_alloc(size);
	hess = gsl_matrix_alloc(size, size);

	// Declaring Eigen map objects to wrap the GSL ones.
	Eigen::Map<Eigen::VectorXd> grad_eigen(grad->data, size);
	Eigen::Map<Eigen::VectorXd> dz_eigen(dz->data, size);
	Eigen::Map<Eigen::MatrixXd> hess_eigen(hess->data, size, size);

	// If epsilon > 0, use its value.  Otherwise, set to machine precision.
	if (epsilon == 0) epsilon = DBL_EPSILON;

	// Initializes the starting value of old_norm at effectively infinity.
	old_norm = 1 / DBL_EPSILON;

	// Start the loop parameter at zero.
	iterations = 0;

	/* If the loop ends and this boolean has not been flipped, the program will
	 * know it has not converged.
	 */
	converged = false;

	// Initializes the value of eta.
	eta = eta_0;

	// Begins the algorithm's loop.
	while (iterations < max_iterations) {

		// Gets the energy, gradient and Hessian for this iteration.
		old_energy = get_energy(size, state);
		get_grad(grad, size, state);
		get_hess(hess, size, state);

		// Adds eta along the diagonal of the Hessian for non-multiplier entries.
		for (unsigned i = 0; i < params; i++) {
			gsl_matrix_set(hess, i, i, gsl_matrix_get(hess, i, i) + eta);
		}

		// Use LU decomposition to solve for the next step in Newton's method.
		dz_eigen = hess_eigen.lu().solve(grad_eigen);

		// Dots the gradient into the step in order to judge the step size.
		gsl_blas_ddot(grad, dz, &grad_dz_prod);

		// Initializes the Armijo counter.
		m = 0;

		// This loop determines the Armijo step size.
		while (true) {
			alpha = gsl_sf_pow_int(beta, m) * s;
			gsl_vector_scale(dz, alpha);
			gsl_vector_sub(state, dz);

			energy = get_energy(size, state);

			if (fabs(old_energy - energy) >= sigma * alpha * grad_dz_prod) break;
			else {
				gsl_vector_add(state, dz);
				gsl_vector_scale(dz, 1 / alpha);
				m++;
			}
		}

		// Gets the new norm of the gradient for comparison.
		norm = gsl_blas_dnrm2(grad) / size;

		// Judges if the norm has changed sufficiently little to decrement eta.
		if (fabs(norm - old_norm) < gamma * eta) eta *= delta;

		// Prints several useful statistics for debugging purposes.
		if (verbose) printf("NEWTON STEP %06d: m %i, grad_norm %e, eta %e, energy %e\n",
				iterations, m, norm, eta, energy);

		// Determines if the process has converged to acceptable precision.
		if (norm < epsilon) {
			converged = true;
			break;
		}

		// Causes the program to fail if norm has diverged to a large number.
		if (norm > bound) break;

		// Reset the norm for the next iteration.
		old_norm = norm;

		if (save_states) {
			char str[40];
			sprintf(str, "states/state-%06d.dat", iterations);
			FILE *fout = fopen(str, "w");
			gsl_vector_fprintf(fout, state, "%.15e");
			fclose(fout);
		}

		// Increment the counter.
		iterations++;
	}

	// Gotta live free, die hard.  No one likes memory leaks.
	gsl_vector_free(grad);
	gsl_vector_free(dz);
	gsl_matrix_free(hess);

	// Return conditions to indicate success or failure.
	if (converged) return 0;
	else return 1;
}

#endif
