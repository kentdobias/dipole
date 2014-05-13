/* domain_minimize.cpp
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

// A Newton's method solver for modulated domains.

// GSL includes.
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

// Gives the necessary functions for the Lagrangian, gradient, and Hessian.
#include "domain_energy.h"
#include "domain_newton.h"

struct nakedgetgrad {
	nakedgetgrad(double c, unsigned n): c(c), n(n) {}
	void operator()(gsl_vector* grad, unsigned size, gsl_vector *state) {domain_energy_nakedGradient(grad, n, state, c);}

	private:
		double c;
		unsigned n;
};

struct nakedgethess {
	nakedgethess(double c, unsigned n): c(c), n(n) {}
	void operator()(gsl_matrix* hess, unsigned size, gsl_vector *state) {domain_energy_nakedHessian(hess, n, state, c);}

	private:
		double c;
		unsigned n;
};

struct nakedgetenergy {
	nakedgetenergy(double c, unsigned n): c(c), n(n) {}
	double operator()(unsigned size, gsl_vector *state) {return domain_energy_nakedLagrangian(n, state, c);}

	private:
		double c;
		unsigned n;
};

// Carries out Newton's method.
int domain_minimize_naked(gsl_vector *z, unsigned n, double c, double eps, unsigned N, double beta, double s, double sigma, double gamma, double eta0, bool verb) {

	unsigned size = 3 * n + 2;
	unsigned params = 2 * n + 1;
	nakedgetgrad grad(c, n);
	nakedgethess hess(c, n);
	nakedgetenergy energy(c, n);

	return domain_newton(z, size, params, energy, grad, hess, eps, N, beta, s, sigma, gamma, eta0, 0.1, 100, verb, false);
}

struct fixedgetgrad {
	fixedgetgrad(double c, unsigned n): c(c), n(n) {}
	void operator()(gsl_vector* grad, unsigned size, gsl_vector *state) {domain_energy_fixedGradient(grad, n, state, c);}

	private:
		double c;
		unsigned n;
};

struct fixedgethess {
	fixedgethess(double c, unsigned n): c(c), n(n) {}
	void operator()(gsl_matrix* hess, unsigned size, gsl_vector *state) {domain_energy_fixedHessian(hess, n, state, c);}

	private:
		double c;
		unsigned n;
};

struct fixedgetenergy {
	fixedgetenergy(double c, unsigned n): c(c), n(n) {}
	double operator()(unsigned size, gsl_vector *state) {return domain_energy_fixedLagrangian(n, state, c);}

	private:
		double c;
		unsigned n;
};

// Carries out Newton's method.
int domain_minimize_fixed(gsl_vector *z, unsigned n, double c, double eps, unsigned N, double beta, double s, double sigma) {

	unsigned size = 3 * n + 3;
	unsigned params = 2 * n;
	fixedgetgrad grad(c, n);
	fixedgethess hess(c, n);
	fixedgetenergy energy(c, n);

	return domain_newton(z, size, params, energy, grad, hess, eps, N, beta, s, sigma, 0, 0, 0.1, 10, true, false);
}

struct randgetgrad {
	randgetgrad(double c, unsigned n, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi): c(c), n(n), ord(ord), k(k), a(a), phi(phi) {}
	void operator()(gsl_vector* grad, unsigned size, gsl_vector *state) {domain_energy_nakedRandGradient(grad, n, state, c, ord, k, a, phi);}

	private:
		double c;
		unsigned n;
		unsigned ord;
		const gsl_vector *k;
		const gsl_vector *a;
		const gsl_vector *phi;
};

struct randgethess {
	randgethess(double c, unsigned n, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi): c(c), n(n), ord(ord), k(k), a(a), phi(phi) {}
	void operator()(gsl_matrix* hess, unsigned size, gsl_vector *state) {domain_energy_nakedRandHessian(hess, n, state, c, ord, k, a, phi);}

	private:
		double c;
		unsigned n;
		unsigned ord;
		const gsl_vector *k;
		const gsl_vector *a;
		const gsl_vector *phi;
};

struct randgetenergy {
	randgetenergy(double c, unsigned n, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi): c(c), n(n), ord(ord), k(k), a(a), phi(phi) {}
	double operator()(unsigned size, gsl_vector *state) {return domain_energy_nakedRandLagrangian(n, state, c, ord, k, a, phi);}

	private:
		double c;
		unsigned n;
		unsigned ord;
		const gsl_vector *k;
		const gsl_vector *a;
		const gsl_vector *phi;
};

// Carries out Newton's method.
int domain_minimize_rand(gsl_vector *z, unsigned n, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double eps, unsigned N, double beta, double s, double sigma, double gamma, double bound, bool verb) {

	unsigned size = 3 * n + 2;
	unsigned params = 2 * n + 1;
	randgetgrad grad(c, n, ord, k, a, phi);
	randgethess hess(c, n, ord, k, a, phi);
	randgetenergy energy(c, n, ord, k, a, phi);

	return domain_newton(z, size, params, energy, grad, hess, eps, N, beta, s, sigma, gamma, bound, 0.1, 2, verb, true);
}

struct nakedwellgetgrad {
	nakedwellgetgrad(double c, unsigned n, double w, double s): c(c), n(n), w(w), s(s) {}
	void operator()(gsl_vector* grad, unsigned size, gsl_vector *state) {domain_energy_nakedWellGradient(grad, n, state, c, w, s);}

	private:
		double c;
		double s;
		double w;
		unsigned n;
};

struct nakedwellgethess {
	nakedwellgethess(double c, unsigned n, double w, double s): c(c), n(n), w(w), s(s) {}
	void operator()(gsl_matrix* hess, unsigned size, gsl_vector *state) {domain_energy_nakedWellHessian(hess, n, state, c, w, s);}

	private:
		double c;
		double s;
		double w;
		unsigned n;
};

struct nakedwellgetenergy {
	nakedwellgetenergy(double c, unsigned n, double w, double s): c(c), n(n), w(w), s(s) {}
	double operator()(unsigned size, gsl_vector *state) {return domain_energy_nakedWellLagrangian(n, state, c, w, s);}

	private:
		double c;
		double s;
		double w;
		unsigned n;
};

// Carries out Newton's method.
int domain_minimize_nakedWell(gsl_vector *z, unsigned n, double c, double w, double ss, double eps, unsigned N, double beta, double s, double sigma, double gamma, double eta0, bool verb) {

	unsigned size = 3 * n + 2;
	unsigned params = 2 * n + 1;
	nakedwellgetgrad grad(c, n, w, ss);
	nakedwellgethess hess(c, n, w, ss);
	nakedwellgetenergy energy(c, n, w, ss);

	return domain_newton(z, size, params, energy, grad, hess, eps, N, beta, s, sigma, gamma, eta0, 0.1, 100, verb, false);
}


struct randwellgetgrad {
	randwellgetgrad(double c, unsigned n, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double s): c(c), n(n), ord(ord), k(k), a(a), phi(phi), w(w), s(s) {}
	void operator()(gsl_vector* grad, unsigned size, gsl_vector *state) {domain_energy_randWellGradient(grad, n, state, c, ord, k, a, phi, w, s);}

	private:
		double c;
		double s;
		double w;
		unsigned n;
		unsigned ord;
		const gsl_vector *k;
		const gsl_vector *a;
		const gsl_vector *phi;
};

struct randwellgethess {
	randwellgethess(double c, unsigned n, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double s): c(c), n(n), ord(ord), k(k), a(a), phi(phi), w(w), s(s) {}
	void operator()(gsl_matrix* hess, unsigned size, gsl_vector *state) {domain_energy_randWellHessian(hess, n, state, c, ord, k, a, phi, w, s);}

	private:
		double c;
		double s;
		double w;
		unsigned n;
		unsigned ord;
		const gsl_vector *k;
		const gsl_vector *a;
		const gsl_vector *phi;
};

struct randwellgetenergy {
	randwellgetenergy(double c, unsigned n, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double s): c(c), n(n), ord(ord), k(k), a(a), phi(phi), w(w), s(s) {}
	double operator()(unsigned size, gsl_vector *state) {return domain_energy_randWellLagrangian(n, state, c, ord, k, a, phi, w, s);}

	private:
		double c;
		double s;
		double w;
		unsigned n;
		unsigned ord;
		const gsl_vector *k;
		const gsl_vector *a;
		const gsl_vector *phi;
};

// Carries out Newton's method.
int domain_minimize_randWell(gsl_vector *z, unsigned n, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double ss, double eps, unsigned N, double beta, double s, double sigma, double gamma, double eta0, bool verb) {

	unsigned size = 3 * n + 2;
	unsigned params = 2 * n + 1;
	randwellgetgrad grad(c, n, ord, k, a, phi, w, ss);
	randwellgethess hess(c, n, ord, k, a, phi, w, ss);
	randwellgetenergy energy(c, n, ord, k, a, phi, w, ss);

	return domain_newton(z, size, params, energy, grad, hess, eps, N, beta, s, sigma, gamma, eta0, 0.1, 100, verb, false);
}


struct fixedmingetgrad {
	fixedmingetgrad(double c, unsigned n): c(c), n(n) {}
	void operator()(gsl_vector* grad, unsigned size, gsl_vector *state) {domain_energy_fixedGradient(grad, n, state, c);}

	private:
		double c;
		unsigned n;
};

struct fixedmingethess {
	fixedmingethess(double c, unsigned n): c(c), n(n) {}
	void operator()(gsl_matrix* hess, unsigned size, gsl_vector *state) {domain_energy_fixedHessian(hess, n, state, c);}

	private:
		double c;
		unsigned n;
};

struct fixedmingetenergy {
	fixedmingetenergy(double c, unsigned n): c(c), n(n) {}
	double operator()(unsigned size, gsl_vector *state) {return domain_energy_fixedLagrangian(n, state, c);}

	private:
		double c;
		unsigned n;
};

// Carries out Newton's method.
int domain_minimize_fixedmin(gsl_vector *z, unsigned n, double c, double eps, unsigned N, double beta, double s, double sigma, double gamma, double bound, bool verb) {

	unsigned size = 3 * n + 3;
	unsigned params = 2 * n;
	fixedmingetgrad grad(c, n);
	fixedmingethess hess(c, n);
	fixedmingetenergy energy(c, n);

	return domain_newton(z, size, params, energy, grad, hess, eps, N, beta, s, sigma, gamma, bound, 0.1, 10, verb, false);
}
