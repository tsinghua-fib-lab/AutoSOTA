dnl *******************************************************************
dnl
dnl	$Id: aclocal.m4,v 1.10 2022/11/19 13:45:49 warme Exp $
dnl
dnl	File:	aclocal.m4
dnl	Rev:	e-3
dnl	Date:	09/24/2016
dnl
dnl	Copyright (c) 1998, 2022 by David M. Warme.  This work is
dnl	licensed under a Creative Commons Attribution-NonCommercial
dnl	4.0 International License.
dnl
dnl *******************************************************************
dnl
dnl	Customized macros for GNU autoconf.
dnl
dnl *******************************************************************
dnl
dnl	Modification Log:
dnl
dnl	a-1:	12/20/98	warme
dnl		: Created.
dnl		: Removed -g from the default CFLAGS.
dnl	a-2:	02/28/2001	warme
dnl		: Changes for 3.1 release, including tests for
dnl		:  Floating point precision fixes and checks
dnl		:  for -lpthread.
dnl	b-1:	09/28/2003	warme
dnl		: Upgrades for autoconf version 2.50 and later.
dnl	e-1:	04/14/2015	warme
dnl		: Changes for 5.0 release.
dnl	e-2:	09/05/2016	warme
dnl		: Change notices for 5.1 release.
dnl	e-3:	09/24/2016	warme
dnl		: Remove floating-point checks, since we have solid
dnl		:  fixes for 32 and 64-bit Intel/AMD CPUs.
dnl
dnl *******************************************************************

dnl
dnl AC_CPLEX_CHECK_CREATEPROB
dnl
dnl Later versions of cplex have eliminated CPXloadlp and related calls
dnl that assemble an entire LP instance in one fell swoop.  These later
dnl versions have only ONE method for creating LP problem instances:
dnl the CPXcreateprob() routine creates an empty LP -- zero rows and
dnl zero columns.  If this routine is available, we assume that the
dnl others are not -- and we provide our own replacement for the ones
dnl we use that simply accomplish similar functionality using only the
dnl newer interfaces.
AC_DEFUN(AC_CPLEX_CHECK_CREATEPROB,
[dnl
 AC_CACHE_CHECK(if CPLEX provides CPXcreateprob function,
		ac_cv_cplex_has_createprob,
	[dnl
	cpx_save_LIBS="${LIBS}"
	cpx_save_CPPFLAGS="${CPPFLAGS}"
	BASIC_CPLEX_LIBS="${LIB} ${ac_cv_cplex_lib}"
	[cpxhdrdir="`echo $ac_cv_cplex_header | sed -e 's!/[^/]*$!!'`"]
	CPPFLAGS="${CPPFLAGS} -I${cpxhdrdir}"

	# We use -lpthread even if it is not needed for this link.
	LIBS="${BASIC_CPLEX_LIBS} -lpthread -lm"
	AC_TRY_LINK(AC_CPLEX_CHECK_PTHREAD_INCLUDES dnl
		    AC_CPLEX_INCLUDE_CTYPE_C, dnl
		    AC_CPLEX_CHECK_CREATEPROB_CODE, dnl
		    ac_cv_cplex_has_createprob=yes,
		    ac_cv_cplex_has_createprob=no)
	LIBS=${cpx_save_LIBS}
	CPPFLAGS=${cpx_save_CPPFLAGS}
 ])
])

dnl
dnl Code to test if CPLEX provides the CPXcreateprob() function.
dnl
AC_DEFUN(AC_CPLEX_CHECK_CREATEPROB_CODE,
[
 int		status;
 CPXENVptr	cpx_env;
 CPXLPptr	lp;

	cpx_env = CPXopenCPLEX (&status);
	lp = CPXcreateprob (cpx_env, &status, "xyzzy");
	CPXfreeprob (cpx_env, &lp);
	CPXcloseCPLEX (&cpx_env);
])

dnl
dnl AC_CPLEX_CHECK_LIBPTHREAD_AND_CTYPE - Check for two more or less
dnl independent properties of the CPLEX we are using:
dnl	1. Do we need to use -lpthread when linking CPLEX?
dnl	2. Do we need to use "ctype.c" in order to work around problems
dnl	   with old CPLEX versions referencing stuff that new glibc's
dnl	   do not provide?
dnl
AC_DEFUN(AC_CPLEX_CHECK_LIBPTHREAD_AND_CTYPE,
[dnl
 AC_CACHE_CHECK(if CPLEX needs -lpthread and/or ctype.c,
		ac_cv_cplex_libpthread,
	[ dnl
	cpx_save_LIBS="${LIBS}"
	cpx_save_CPPFLAGS="${CPPFLAGS}"
	BASIC_CPLEX_LIBS="${LIB} ${ac_cv_cplex_lib}"
	[cpxhdrdir="`echo $ac_cv_cplex_header | sed -e 's!/[^/]*$!!'`"]
	CPPFLAGS="${CPPFLAGS} -I${cpxhdrdir}"

	LIBS="${BASIC_CPLEX_LIBS} -lm"
	AC_TRY_LINK(AC_CPLEX_CHECK_PTHREAD_INCLUDES, dnl
		    AC_CPLEX_CHECK_PTHREAD_CODE, dnl
		    dnl Neither -lpthread nor csave.c are needed.
		    ac_cv_cplex_libpthread=no
		    ac_cv_cplex_needs_ctype=no,	dnl
		    # Link did not work.  Try adding -lpthread.
		    LIBS="${BASIC_CPLEX_LIBS} -lpthread -lm"
		    AC_TRY_LINK(AC_CPLEX_CHECK_PTHREAD_INCLUDES, dnl
				AC_CPLEX_CHECK_PTHREAD_CODE, dnl
				dnl We need -lpthread, but not ctype.c
				ac_cv_cplex_libpthread=yes
				ac_cv_cplex_needs_ctype=no,
				# Link did not work.  Try using ctype.c
				# without -lpthread
				LIBS="${BASIC_CPLEX_LIBS} -lm"
				AC_TRY_LINK(AC_CPLEX_CHECK_PTHREAD_INCLUDES dnl
					    AC_CPLEX_INCLUDE_CTYPE_C, dnl
					    AC_CPLEX_CHECK_PTHREAD_CODE, dnl
					    dnl We need ctype.c, but not -lpthread
					    ac_cv_cplex_libpthread=no
					    ac_cv_cplex_needs_ctype=yes,
					    # Link did not work.  Try using
					    # both -lpthread and ctype.c
					    LIBS="${BASIC_CPLEX_LIBS} -lpthread -lm"
					    AC_TRY_LINK(AC_CPLEX_CHECK_PTHREAD_INCLUDES dnl
							AC_CPLEX_INCLUDE_CTYPE_C, dnl
							AC_CPLEX_CHECK_PTHREAD_CODE, dnl
							dnl We need both ctype.c and -lpthread
							ac_cv_cplex_libpthread=yes
							ac_cv_cplex_needs_ctype=yes,
							# None of the 4 combinations
							# worked, so just fall back to the
							# unadorned settings and let the
							# user figure it out!
							AC_MSG_WARN(unable to link with cplex)
							ac_cv_cplex_libpthread=no
							ac_cv_cplex_needs_ctype=no))))
	LIBS=${cpx_save_LIBS}
	CPPFLAGS=${cpx_save_CPPFLAGS}
 ])
])

dnl
dnl Macro to generate proper include files for -lpthread checking code.
dnl
AC_DEFUN(AC_CPLEX_CHECK_PTHREAD_INCLUDES, dnl
[#if CPLEX < 40
 #include "cpxdefs.inc"
#else
 #include "cplex.h"
#endif
])

dnl
dnl Macro to generate an include "ctype.c".
dnl
AC_DEFUN(AC_CPLEX_INCLUDE_CTYPE_C, dnl
[#include "ctype.c"
])

dnl
dnl Code to test if CPLEX needs -lpthread
dnl
AC_DEFUN(AC_CPLEX_CHECK_PTHREAD_CODE,
[
#if CPLEX >= 50
 int		status;
/* CPXENVptr	cpx_env; */
 CPXLPptr	lp;

/*	cplex_env = CPXopenCPLEX (&status); */
	lp = NULL;
	CPXprimopt ((void *) 0, lp);
	CPXdualopt ((void *) 0, lp);
#else
 #if CPLEX >= 40
  int		status;
  CPXENVptr	cpx_env;
  CPXLPptr	lp;

	cplex_env = CPXopenCPLEX (&status);
	lp = NULL;
	CPXoptimize (cpx_env, lp);
	CPXdualopt (cpx_env, lp);
 #else
  CPXLPptr	lp;

	lp = NULL;
	optimize (lp);
	dualopt (lp);
 #endif
#endif
])

dnl AC_DEFINE_EXPAND_VALUE(VARIABLE [, VALUE])
AC_DEFUN(AC_DEFINE_EXPAND_VALUE,
[dnl call AC_DEFINE after expanding given value
ac_expand_text1=""
ac_expand_text2="`eval echo \"$2\"`"
while test "$ac_expand_text1" != "$ac_expand_text2"
do
	ac_expand_text3="`eval echo \"$ac_expand_text2\"`"
	ac_expand_text1="$ac_expand_text2"
	ac_expand_text2="$ac_expand_text3"
done
AC_DEFINE_UNQUOTED([$1], "$ac_expand_text1")
])

dnl AC_FIND_FILE(VARIABLE, FILENAME, TESTFLAG, DIRS)
AC_DEFUN(AC_FIND_FILE,
[dnl loop over the dirs, until found
  $1=''
  for cv_ff_dir in $4
  do
	if test "$3" "$cv_ff_dir/$2"
	then
		$1="$cv_ff_dir/$2"
		break
	fi
  done
])

dnl AC_GMP_WORKS_NORMALLY(ACTION_IF_TRUE,ACTION_IF_FALSE)
AC_DEFUN(AC_GMP_WORKS_NORMALLY,
[dnl Check to see if GMP works using #include <gmp.h> and -lgmp -lm
AC_CACHE_CHECK(to see if GMP installed and works,
		ac_cv_gmp_installed_and_works,
		AC_LANG_SAVE
		AC_LANG_C
		ac_save_LIBS="$LIBS"
		LIBS="$LIBS -lgmp"
		AC_TRY_RUN([AC_GMP_CODE],
			   ac_cv_gmp_installed_and_works=yes,
			   ac_cv_gmp_installed_and_works=no,
			   ac_cv_gmp_installed_and_works=no)
		LIBS="$ac_save_LIBS"
		AC_LANG_RESTORE)
if test "$ac_cv_gmp_installed_and_works" = yes
then
	$1
else
	$2
fi
])

AC_DEFUN(AC_GMP_CODE,
[#include <gmp.h>
int main (int argc, char **argv)
{
int	i, j;
mpf_t	X;
mpq_t	A, B, C;
mpz_t	K, N, M;
mpz_ptr	p;
double	z;

	/* mpf functions */
	mpf_init2 (X, 64);
	mpf_set_d (X, 3.14159265358979323);
	mpf_clear (X);

	/* mpz functions */

	mpz_init_set_ui (N, 1);
	for (i = 2; i <= 102; i++) {
		mpz_mul_ui (N, N, i);
	}
	mpz_mul_2exp (N, N, 91);

	mpz_init_set_ui (M, 1);
	for (i = 2; i <= 100; i++) {
		mpz_mul_ui (M, M, i);
	}
	mpz_mul_2exp (M, M, 89);
	mpz_init_set_ui (K, 2);

	/* mpq functions */
	mpq_init (A);
	mpq_init (B);
	mpq_set_ui (A, 1, 1);
	p = mpq_numref (A);
	mpz_mul (p, p, N);
	mpz_set_ui (mpq_denref (A), 1);
	mpz_mul (mpq_denref (A), mpq_denref (A), M);
	mpq_canonicalize (A);
	mpz_sub_ui (K, mpq_numref (A), 4*101*102);
	if (mpz_sgn (K) != 0) {
		return 1;
	}
	mpz_sub_ui (K, mpq_denref (A), 1);
	if (mpz_sgn (K) != 0) {
		return 2;
	}
	mpz_set (K, N);
	mpz_mul (K, K, N);

	mpq_set (B, A);

	for (i = 2; i <= 5; i++) {
		mpq_mul (B, B, B);
	}
	if (mpq_cmp (A, B) >= 0) {
		return 4;
	}
	mpq_div (B, B, B);
	z = mpq_get_d (B);
	if (z != 1.0) {
		return 3;
	}
	mpq_add (B, B, A);
	mpq_sub (A, B, B);
	if (mpz_sgn (mpq_numref (A)) != 0) {
		return 5;
	}

	mpq_clear (B);
	mpq_clear (A);

	mpz_clear (K);
	mpz_clear (M);
	mpz_clear (N);

	return 0;
}
])
