/* domain_init.cpp
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

// Initializes a circular domain, or a circular domain with a perturbation.

#include <unistd.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>

// GSL includes.
#include <gsl/gsl_sf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>

#include "domain_minimize.h"
#include "domain_energy.h"




int main(int argc, char *argv[]) {

	// Declaring variables.
	gsl_vector *z, *new_z;
	int opt;
	unsigned n;
	double c;
	char out_file[20];

	// Default values.
	sprintf(out_file, "%s", "out.dat");

	// GNU getopt in action.
	while ((opt = getopt(argc, argv, "n:c:i:")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'c':
				c = atof(optarg);
				break;
			case 'i':
				sprintf(out_file, "%s", optarg);
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}

	z = gsl_vector_alloc(3 * n + 3);
	new_z = gsl_vector_alloc(6 * n + 3);

	FILE *f = fopen(out_file, "r");
	gsl_vector_fscanf(f, z);
	fclose(f);

	gsl_vector_set(new_z, 2 * n - 1, (gsl_vector_get(z, 0) + gsl_vector_get(z, n - 1)) / 2);
		gsl_vector_set(new_z, 2 * n - 2, gsl_vector_get(z, n - 2));

	for (unsigned i = 0; i < n - 1; i ++) {
		gsl_vector_set(new_z, 2 * i, gsl_vector_get(z, i));
		gsl_vector_set(new_z, 2 * i + 1,
				(gsl_vector_get(z, i) + gsl_vector_get(z, i + 1)) / 2);
	}

	gsl_vector_set(new_z, 2 * n, gsl_vector_get(z, n) / 2);
	gsl_vector_set(new_z, 2 * n + 1, gsl_vector_get(z, n));
	gsl_vector_set(new_z, 4 * n - 2, gsl_vector_get(z, 2 * n - 2) / 2);
	gsl_vector_set(new_z, 4 * n - 3, gsl_vector_get(z, 2 * n - 2));

	for (unsigned i = 1; i < n - 1; i ++) {
		gsl_vector_set(new_z, 2 * (n + i) + 1, gsl_vector_get(z, n + i));
		gsl_vector_set(new_z, 2 * (n + i) ,
				(gsl_vector_get(z, n + i - 1) + gsl_vector_get(z, n + i)) / 2);
	}

	gsl_vector_set(new_z, 4 * n - 1, gsl_vector_get(z, 2 * n -1));

	gsl_vector_set(new_z, 4 * n, gsl_vector_get(z, 2 * n));

	for (unsigned i = 0; i < n; i++) {
		gsl_vector_set(new_z, 4 * n + 1 + 2 * i, gsl_vector_get(z, 2 * n + 1 + i));
		gsl_vector_set(new_z, 4 * n + 2 + 2 * i, gsl_vector_get(z, 2 * n + 1 + i));
	}

	gsl_vector_set(new_z, 6 * n + 1, gsl_vector_get(z, 3 * n + 1));
	gsl_vector_set(new_z, 6 * n + 2, gsl_vector_get(z, 3 * n + 2));

	int result;
	result = domain_minimize_fixed(new_z, 2 * n, c, 1e-8, 0.00001, 5000, 4, 2);

	if (result) printf("Converging failed.");
	else {
		FILE *fout = fopen(out_file, "w");
		gsl_vector_fprintf(fout, new_z, "%.10g");
		fclose(fout);
	}

	gsl_vector_free(z);
	gsl_vector_free(new_z);

}

