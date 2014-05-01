#ifndef DOMAIN_MINIMIZE_H
#define DOMAIN_MINIMIZE_H

// GSL includes.
#include <gsl/gsl_vector.h>

// Gives the necessary functions for the Lagrangian, gradient, and Hessian.
#include "domain_energy.h"
#include "domain_newton.h"

int domain_minimize_naked(gsl_vector *z, unsigned n, double c, double eps, unsigned N, double beta, double s, double sigma, double gamma, double eta0, bool verb);

int domain_minimize_fixed(gsl_vector *z, unsigned n, double c, double eps, unsigned N, double beta, double s, double sigma);

int domain_minimize_rand(gsl_vector *z, unsigned n, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double eps, unsigned N, double beta, double s, double sigma, double gamma, double bound, bool verb);

int domain_minimize_nakedWell(gsl_vector *z, unsigned n, double c, double w, double ss, double eps, unsigned N, double beta, double s, double sigma, double gamma, double bound, bool verb);

int domain_minimize_randWell(gsl_vector *z, unsigned n, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double ss, double eps, unsigned N, double beta, double s, double sigma, double gamma, double bound, bool verb);

int domain_minimize_fixedmin(gsl_vector *z, unsigned n, double c, double eps, unsigned N, double beta, double s, double sigma, double gamma, double bound, bool verb);

#endif

