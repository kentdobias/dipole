#ifndef DOMAIN_EIGEN_H
#define DOMAIN_EIGEN_H

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

void domain_eigen_values(gsl_vector *eigenvalues, unsigned size, unsigned params,
		gsl_matrix *hess);

void domain_eigen_sort(gsl_permutation *eigenorder, unsigned size, unsigned eigen_num,
		const gsl_vector *eigenvalues);

void domain_eigen_state(gsl_vector *eigenstate, const gsl_vector *eigenvalues,
		unsigned n, double thres);

void domain_eigen_vector(gsl_vector *eigenvector, unsigned size, unsigned params, unsigned k, gsl_matrix *hess);

void domain_eigen_ortho(gsl_vector *eigenvector, unsigned n, const gsl_vector *z);

#endif
