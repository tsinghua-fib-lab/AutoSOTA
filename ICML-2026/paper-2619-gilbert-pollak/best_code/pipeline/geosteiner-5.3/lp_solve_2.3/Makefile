# Converted to use GNU autoconf configure script by warme 01/11/99.
# Added "dist" target by warme 04/14/2015.

# Make support

SHELL = /bin/sh


# Various Programs and options we use:

CC	= gcc
CFLAGS	= -O3
YACC	= yacc
LEX	= :
LEXLIB	= 
RANLIB	= ranlib

#should be OK in most situations:
#CFLAGS= -O
#
# HP/UX 9.0X optimized code
#CFLAGS= +O3 +Oaggressive +Olibcalls -Aa -D_POSIX_SOURCE -DCHECK +FP VZOUiD
# HP/UX 9.0X debugging
#CFLAGS= -g -Aa -D_POSIX_SOURCE -DCHECK +FP VZOUiD
#
# nice for gcc
#CFLAGS= -O3 -Wall -pedantic -ansi
#CFLAGS= -g -Wall -pedantic -ansi

# Option -DCHECK checks for numerical problems during rounding of numbers.
# It will slow things down a bit.
# You can add a -DREAL=<float type> to the CFLAGS, to change the default float
# type used in lp_solve (double) to float or 'long double'. However, type float
# might be fast on your computer, but it is not accurate enough to solve even
# moderately sized problems without running into numerical problems.
# The use of long doubles does increase the numerical stability of lp_solve,
# if your compiler actually implements them with more bits than a double. But
# it slows down things quite a bit.

#ANSI math lib
#MATHLIB= -lM
#non-ANSI math lib, should also work
MATHLIB= -lm

LPKSRC.c= lpkit.c solve.c debug.c read.c readmps.c hash.c presolve.c lpbinio.c
LEXFILE.l= lex.l
YACCFILE.y= lp.y
TESTFILES= lp_examples/ex1.lp lp_examples/ex2.lp lp_examples/ex3.lp lp_examples/ex4.lp lp_examples/ex5.lp lp_examples/ex6.lp lp_examples/ex7.lp

# libLPS.a contains only the stuff needed by GeoSteiner.

TARGET	= lp_solve
LPKLIB	= liblpk.a
LPSLIB	= libLPS.a

LEXFILE.c= $(LEXFILE.l:.l=.c)
YACCFILE.c= $(YACCFILE.y:.y=.c)
YACCFILE.o= $(YACCFILE.y:.y=.o)
CSOURCES=lpkit.c solve.c debug.c read.c readmps.c lp_solve.c demo.c hash.c presolve.c $(LEXFILE.c) $(YACCFILE.c) lpbinio.c
COBJ=$(CSOURCES:.c=.o)
LPKSRC= $(LPKSRC.c) $(YACCFILE.c)
LPKOBJ= $(LPKSRC:.c=.o)

# Files needed by GeoSteiner:
LPSSRC = lpkit.c solve.c debug.c presolve.c hash.c lpbinio.c
LPSOBJ = $(LPSSRC:.c=.o)

HEADERS=lpkit.h lpglob.h patchlevel.h debug.h read.h hash.h

all:	demo $(TARGET) lp2mps mps2lp $(LPSLIB)

$(COBJ): $(HEADERS)

demo:	demo.o $(LPKLIB)
	$(CC) -o demo $(CFLAGS) demo.o $(LPKLIB) $(LEXLIB) $(MATHLIB)

lp2mps:	lp2mps.o $(LPKLIB)
	$(CC) -o lp2mps $(CFLAGS) lp2mps.o $(LPKLIB) $(LEXLIB) $(MATHLIB)

mps2lp: mps2lp.o $(LPKLIB)
	$(CC) -o mps2lp $(CFLAGS) mps2lp.o $(LPKLIB) $(LEXLIB) $(MATHLIB)

$(TARGET): lp_solve.o $(LPKLIB)
	$(CC) -o $(TARGET) $(CFLAGS) lp_solve.o $(LPKLIB) $(LEXLIB) $(MATHLIB)

$(LPKLIB): $(LPKOBJ)
	ar rv $@ $(LPKOBJ)
	$(RANLIB) $(LPKLIB)

$(LPSLIB): $(LPSOBJ)
	ar rv $@ $(LPSOBJ)
	$(RANLIB) $(LPSLIB)

$(YACCFILE.o): $(LEXFILE.c)

$(LEXFILE.c): $(LEXFILE.l)
	$(LEX) $(LEXFILE.l)
	mv lex.yy.c $(LEXFILE.c)

$(YACCFILE.c): $(YACCFILE.y)
	$(YACC) $(YACCFILE.y)
	mv y.tab.c $(YACCFILE.c)

test:
	-for i in $(TESTFILES); do\
		./$(TARGET) -p -s -time < $$i > xxx.tmp;\
		if diff xxx.tmp lp_examples/`basename $$i .lp`.out > /dev/null; then\
			echo "$$i gives the correct result";\
		else\
			echo "*** $$i gives different result, please check ***";\
		fi;\
	done;\
	rm xxx.tmp

$(TARGET).man: $(TARGET).1
	nroff -man $(TARGET).1 > $(TARGET).man

clean:
	rm -f *.a *.o TAGS $(LEXFILE.c) $(YACCFILE.c) demo $(TARGET) lp2mps mps2lp .pure .softdebughist datafile

distclean : clean
	-rm -f Makefile config.cache config.h config.log config.status
#

main_files_to_distribute := \
	CHANGELOG \
	debug.c \
	debug.h \
	demo.c \
	HARTMUT_DOCUMENTATION \
	hash.c \
	hash.h \
	lex.l \
	lp2mps.c \
	lpbinio.c \
	lpglob.h \
	lpkit.c \
	lpkit.h \
	lp_solve.1 \
	lp_solve.c \
	lp_solve.man \
	lp.y \
	Makefile.in \
	MANIFEST \
	MIPLIB_RESULTS \
	mps2lp.c \
	MPS.description \
	NETLIB_RESULTS \
	patchlevel.h \
	presolve.c \
	read.c \
	read.h \
	README \
	README.custom \
	readmps.c \
	solve.c

examples_to_distribute := \
	demo_lag.lp \
	demo_lag.out \
	ex1.lp \
	ex1.mps \
	ex1.out \
	ex2.lp \
	ex2.out \
	ex3.lp \
	ex3.out \
	ex4.lp \
	ex4.out \
	ex5.lp \
	ex5.out \
	ex6.lp \
	ex6.out \
	ex7.lp \
	ex7.out

# Making distribution tarball from GeoSteiner.

dist:
	@if test "$(distdir)" = ""; \
	then \
		echo "lp_solve_2.3/Makefile: dist without \$$(distdir)!"; \
		exit 1; \
	fi
	@if test -e "$(distdir)"; \
	then \
		echo "lp_solve_2.3/Makefile: distdir already exists!"; \
		exit 1; \
	fi
	@echo Copying files...
	@mkdir $(distdir) $(distdir)/lp_examples
	@chmod 755 $(distdir) $(distdir)/lp_examples
	@cp $(main_files_to_distribute) $(distdir)
	@for i in $(main_files_to_distribute); \
	do \
		touch -r $$i $(distdir)/$$i; \
	done
	@(cd $(distdir) && chmod 644 $(main_files_to_distribute))
	@(cd lp_examples && cp $(examples_to_distribute) $(distdir)/lp_examples)
	@for i in $(examples_to_distribute); \
	do \
		touch -r lp_examples/$$i $(distdir)/lp_examples/$$i; \
	done
	@(cd $(distdir)/lp_examples && chmod 644 $(examples_to_distribute))
