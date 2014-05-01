#ifndef DOMAIN_ENERGY_H
#define DOMAIN_ENERGY_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

double domain_energy_nakedEnergy(unsigned n, const gsl_vector *z, double c);

double domain_energy_nakedLagrangian(unsigned n, const gsl_vector *z, double c);

double domain_energy_nakedGradient(gsl_vector *grad, unsigned n, const gsl_vector *z, double c);

double domain_energy_nakedHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c);

double domain_energy_fixedEnergy(unsigned n, const gsl_vector *z, double c);

double domain_energy_fixedLagrangian(unsigned n, const gsl_vector *z, double c);

double domain_energy_fixedGradient(gsl_vector *grad, unsigned n, const gsl_vector *z, double c);

double domain_energy_fixedHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c);

double domain_energy_nakedRandLagrangian(unsigned n, const gsl_vector *z, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi);

double domain_energy_nakedRandGradient(gsl_vector *grad, unsigned n, const gsl_vector *z, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi);

double domain_energy_nakedRandHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi);

double domain_energy_nakedWellLagrangian(unsigned n, const gsl_vector *z, double c, double w, double s);

double domain_energy_nakedWellGradient(gsl_vector *grad, unsigned n, const gsl_vector *z, double c, double w, double s);

double domain_energy_nakedWellHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c, double w, double s);

double domain_energy_randWellLagrangian(unsigned n, const gsl_vector *z, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double s);

double domain_energy_randWellGradient(gsl_vector *grad, unsigned n, const gsl_vector *z, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double s);

double domain_energy_randWellHessian(gsl_matrix *hess, unsigned n, const gsl_vector *z, double c, unsigned ord, const gsl_vector *k, const gsl_vector *a, const gsl_vector *phi, double w, double s);

#endif
