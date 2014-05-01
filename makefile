# domain_tools - modulated domain toolkit
# See COPYING file for copyright and license details.

CC = g++
CFLAGS = -Ofast -std=c++11 -fopenmp
LDFLAGS = -lgslcblas -lgsl -latlas

OBJ = domain_energy domain_minimize domain_eigen
BIN = bifurchaser initialize domain_increase domain_improve hessget gradget evolve eigenvalues

all: opt ${OBJ:%=obj/%.o} ${BIN:%=bin/%}

opt:
	@echo build options:
	@echo "CC       = ${CC}"
	@echo "CFLAGS   = ${CFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"

obj/%.o: src/%.cpp
	@echo CC -c -o $@
	@${CC} ${CFLAGS} ${LDFLAGS} -c -o $@ $<

bin/%: src/%.cpp ${OBJ:%=obj/%.o}
	@echo CC -o $@
	@${CC} ${COBJ:%=obj/%.o} ${OBJ:%=obj/%.o} ${CFLAGS} ${LDFLAGS} -o $@ $<

clean:
	@echo cleaning:
	@echo rm -f ${OBJ:%=obj/%.o} ${BIN:%=bin/%}
	@rm -f ${OBJ:%=obj/%.o} ${BIN:%=bin/%}

.PHONY: all clean
