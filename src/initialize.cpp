/* initialize.cpp
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

// Initializes modulated domains.

#include <unistd.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <gsl/gsl_sf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>


int main(int argc, char *argv[]) {

	// Declaring variables.
	gsl_vector *z, *k, *a, *phi;
	int opt;
	unsigned n, m, size, params, word, ord;
	double p, k0, a0, R, w0, slope;
	char *out_filename, *k_filename, *a_filename, *phi_filename;
	bool rand;

	// Default values.
	n = 100;
	p = 0;
	m = 0;
	rand = false;
	k0 = 1;
	a0 = 1;
	ord = 0;
	word = 0;
	w0=1;
	slope=1;

	// GNU getopt in action.
	while ((opt = getopt(argc, argv, "n:p:m:o:O:rK:A:P:k:a:wW:u:s:")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'p':
				p = atof(optarg);
				break;
			case 'a':
				a0 = atof(optarg);
				break;
			case 'k':
				k0 = atof(optarg);
				break;
			case 'm':
				m = atoi(optarg);
				break;
			case 'O':
				ord = atoi(optarg);
				break;
			case 'o':
				out_filename = optarg;
				break;
			case 'K':
				k_filename = optarg;
				break;
			case 'A':
				a_filename = optarg;
				break;
			case 'P':
				phi_filename = optarg;
				break;
			case 'r':
				rand = true;
				break;
			case 'W':
				word = atoi(optarg);
				break;
			case 'u':
				w0 = atof(optarg);
				break;
			case 's':
				slope = atof(optarg);
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}

	if (rand) {
		size = 3 * n + 2;
		params = 2 * n + 1;
	} else {
		size = 3 * n + 3;
		params = 2 * n;
	}

	z = gsl_vector_alloc(size);
	if (rand) {
		k = gsl_vector_alloc(2 * (ord + 2 * word));
		a = gsl_vector_alloc(ord + 2 * word);
		phi = gsl_vector_alloc(ord + 2* word);
	}

	R = sqrt(2 * M_PI / (n * sin(2 * M_PI / n)));

	// setting x[0..n], y[1..n]
	for (unsigned i = 0; i < n; i++) {
		gsl_vector_set(z, i, R * cos(2 * M_PI * i / n));
		if (rand) gsl_vector_set(z, n + i, R * sin(2 * M_PI * i / n));
		else if (i != 0) gsl_vector_set(z, n + i - 1, R * sin(2 * M_PI * i / n));
	}

	// setting L
	gsl_vector_set(z, params - 1, 2 * R * n * sin(M_PI / n));

	for (unsigned i = 0; i < size - params; i++) {
		gsl_vector_set(z, params + i, 1);
	}

	if (p > 0) {
		for (unsigned i = 0; i < n; i++) {
			gsl_vector_set(z, i, gsl_vector_get(z, i) *
					(1 + p * cos(2 * M_PI * m * i / n)));

			if (rand) gsl_vector_set(z, n + i, gsl_vector_get(z, n + i) *
					(1 + p * cos(2 * M_PI * m * i / n)));
			else if (i != 0) gsl_vector_set(z, n + i - 1, gsl_vector_get(z, n + i - 1) *
					(1 + p * cos(2 * M_PI * m * i / n)));
		}
	}
	if (rand) {
		// The GSL random number generator.  Don't try to make more than one random
		// background in the same second, or you'll get identical results.
		gsl_rng *rand;
		rand = gsl_rng_alloc(gsl_rng_ranlux);
		gsl_rng_set(rand, time(NULL));

		double kx, ky, kk;

		for (unsigned i = 0; i < ord; i++) {
			while (true) {
				kx = gsl_rng_uniform(rand) * 2 - 1;
				ky = gsl_rng_uniform(rand) * 2 - 1;
				kk = gsl_pow_2(kx)+gsl_pow_2(ky);

				if (kk < 1) {
					gsl_vector_set(k, i, k0 * kx);
					gsl_vector_set(k, i + ord + 2 * word, k0 * ky);
					break;
				}
			}
			gsl_vector_set(a, i, 2 * a0 / ord * gsl_rng_uniform(rand));
			gsl_vector_set(phi, i, 2 * M_PI * gsl_rng_uniform(rand));
		}
	}

	if (rand) {
		FILE *k_file = fopen(k_filename, "w");
		gsl_vector_fprintf(k_file, k, "%.10g");
		fclose(k_file);

		FILE *a_file = fopen(a_filename, "w");
		gsl_vector_fprintf(a_file, a, "%.10g");
		fclose(a_file);

		FILE *phi_file = fopen(phi_filename, "w");
		gsl_vector_fprintf(phi_file, phi, "%.10g");
		fclose(phi_file);
	}

	FILE *out_file = fopen(out_filename, "w");
	gsl_vector_fprintf(out_file, z, "%.10g");
	fclose(out_file);


	gsl_vector_free(z);
	if (rand) {
		gsl_vector_free(phi);
		gsl_vector_free(k);
		gsl_vector_free(a);
	}

}

