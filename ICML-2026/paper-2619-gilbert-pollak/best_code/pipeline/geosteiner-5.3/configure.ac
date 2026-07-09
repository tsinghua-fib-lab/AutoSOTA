dnl ******************************************************************
dnl
dnl	$Id: configure.ac,v 1.18 2023/04/03 16:30:01 warme Exp $
dnl
dnl	File:	configure.ac
dnl	Rev:	e-7
dnl	Date:	04/03/2023
dnl
dnl	Copyright (c) 1998, 2023 by David M. Warme.  This work is
dnl	licensed under a Creative Commons Attribution-NonCommercial
dnl	4.0 International License.
dnl
dnl ******************************************************************
dnl
dnl	Process this file with autoconf to produce a configure script
dnl	for GeoSteiner.
dnl
dnl ******************************************************************
dnl
dnl	Modification Log:
dnl
dnl	a-1:	12/20/98	warme
dnl		: Created.
dnl	a-2:	08/07/97	warme
dnl		: Added check for bison.  This doesn't really work
dnl		:  the way we would like it to.  We would prefer the
dnl		:  configuration to FAIL if no bison can be found!
dnl	b-1:	02/28/2001	warme
dnl		: Changes for 3.1 release.  Added checks for -lpthread,
dnl		:  -ffloat-store and Intel floating point precision
dnl		:  fix, GMP, and stderr being an lvalue.
dnl	b-2:	09/28/2003	warme
dnl		: Upgraded to autoconf version 2.53 and later.
dnl		: Changed name from configure.in to configure.ac.
dnl	d-1:	02/25/2006     warme
dnl		: Removed geogui.
dnl	e-1:	04/14/2015	warme
dnl		: Changes for 5.0 release.
dnl		: Add AC_SUBST() for all pieces of version number.
dnl		: Provide CPLEX header/library directory vars.
dnl		: Remove ability to disable efst, rfst, ufst, pruning.
dnl		: Remove commercial stuff.
dnl	e-2:	12/12/2015	warme
dnl		: Version 5.0.1.
dnl	e-3:	09/05/2016	warme
dnl		: Version 5.1.
dnl		: Fix libtool for OSX/Darwin.
dnl		: Change notices for 5.1 release.
dnl	e-4:	09/24/2016	warme
dnl		: Remove floating-point precision checks.
dnl	e-5:	09/30/2016	warme
dnl		: One more try to fix libtool on OSX/Darwin.
dnl	e-6:	11/18/2022	warme
dnl		: Version 5.2.
dnl		: Changed default CFLAGS from -O2 to -O3.
dnl		: Added --with-Werror to enable -Werror flag.
dnl	e-7:	04/03/2023	warme
dnl		: Version 5.3.
dnl
dnl ******************************************************************
dnl
AC_INIT(steiner.h)

AC_SUBST(CFLAGS)

dnl This is where you set the software version number!
GEOLIB_VERSION_MAJOR=5
GEOLIB_VERSION_MINOR=3
GEOLIB_VERSION_PATCH=0
GEOLIB_VERSION_STRING="${GEOLIB_VERSION_MAJOR}.${GEOLIB_VERSION_MINOR}.${GEOLIB_VERSION_PATCH}"

AC_DEFINE_UNQUOTED(GEOLIB_VERSION_MAJOR, $GEOLIB_VERSION_MAJOR)
AC_DEFINE_UNQUOTED(GEOLIB_VERSION_MINOR, $GEOLIB_VERSION_MINOR)
AC_DEFINE_UNQUOTED(GEOLIB_VERSION_PATCH, $GEOLIB_VERSION_PATCH)
AC_DEFINE_UNQUOTED(GEOLIB_VERSION_STRING, "$GEOLIB_VERSION_STRING")

AC_SUBST(GEOLIB_VERSION_MAJOR)
AC_SUBST(GEOLIB_VERSION_MINOR)
AC_SUBST(GEOLIB_VERSION_PATCH)
AC_SUBST(GEOLIB_VERSION_STRING)

AC_PREFIX_DEFAULT(/usr/local)

AC_CONFIG_HEADER(config.h)

dnl We want the default value of CFLAGS to be "-O3" instead of "-g" or
dnl "-g -O2" or something else.
test "${CFLAGS+set}" = set || CFLAGS='-O3'

dnl Checks for programs.
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_INSTALL
AC_PROG_YACC
AC_PROG_LEX
AC_PROG_MAKE_SET

dnl Make sure we have maximum compatibility with lex when using flex!
if test "$LEX" = "flex"; then LEX='flex -l'; fi

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIGNAL

dnl Check for system header files we could use...
AC_CHECK_HEADERS(sys/utsname.h)
AC_HEADER_STDC

dnl Determine whether Unix CPU time stuff is available.
AC_CACHE_CHECK(for times() and CLK_TCK,
	ac_cv_header_unix_cpu_time,
AC_TRY_COMPILE(
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <time.h>
#include <unistd.h>
, clock_t		ticks = sysconf (_SC_CLK_TCK);
  clock_t		total;
  struct tms		t;

	times (&t);
	total = t.tms_utime + t.tms_stime + t.tms_cutime + t.tms_cstime;
	
, ac_cv_header_unix_cpu_time=yes, ac_cv_header_unix_cpu_time=no))

if test $ac_cv_header_unix_cpu_time = yes; then
	AC_DEFINE(UNIX_CPU_TIME)
fi

dnl Determine whether the uname function works.
AC_CACHE_CHECK(for working uname function,
	ac_cv_func_uname_works,
AC_TRY_COMPILE(
#include <sys/utsname.h>
, struct utsname	un;
  int			total;

	uname (&un);
	total = *un.sysname
	      + *un.nodename
	      + *un.release
	      + *un.version
	      + *un.machine;
	printf ("%d\n", total);
, ac_cv_func_uname_works=yes, ac_cv_func_uname_works=no))

if test $ac_cv_func_uname_works = yes; then
	AC_DEFINE(UNAME_FUNCTION_WORKS)
fi

dnl Check for some functions we would like to use, if available.
AC_CHECK_FUNCS(fsync link rename sync unlink)

dnl Check if popen and pclose are available.
AC_CHECK_FUNCS(popen pclose)

dnl Check for the uname command.
AC_PATH_PROG(ac_cv_prog_uname_full_pathname, uname)

if test -n "$ac_cv_prog_uname_full_pathname"
then
  AC_DEFINE_UNQUOTED(UNAME_PATH, "$ac_cv_prog_uname_full_pathname")
fi

dnl We need libtool, but it is called glibtool on OSX/Darwin.
AC_PATH_PROGS(LIBTOOL, glibtool /usr/local/bin/glibtool libtool, foo)
if test "${LIBTOOL}" = "foo"
then
	# Can't find libtool -- abort here!
	AC_MSG_ERROR([GeoSteiner needs 'libtool' and cannot find it.], 1)
fi

AC_SUBST(WERROR_FLAG)

dnl See if the user has specified --with-Werror.
AC_ARG_WITH(Werror,
[	--with-Werror         Uses both -Wall and -Werror],
if test "$withval" = no
then
	WERROR_FLAG=''
else
	if test "$withval" != yes
	then
		echo "Arg to --with-Werror must be yes or no!" 1>&2
		exit 1
	fi
	WERROR_FLAG='-Werror'
fi
)

dnl Check if stderr can be used as an lvalue...
AC_CACHE_CHECK(if stderr is an lvalue, ac_cv_stderr_is_lvalue,
  AC_TRY_LINK([#include <stdio.h>
], [
	FILE *		fp1;
	FILE *		fp2;
	fp1 = stderr;
	fp2 = fopen ("/dev/null", "w");
	stderr = fp2;

	fprintf (stderr, "Discarded output\n");

	stderr = fp1;
	fclose (fp2);
],
 ac_cv_stderr_is_lvalue=yes,
   ac_cv_stderr_is_lvalue=no))

if test "$ac_cv_stderr_is_lvalue" = "yes"
then
	AC_DEFINE(HAVE_STDERR_IS_LVALUE)
fi

dnl Determine whether we have sigaction() function and "struct sigaction".
AC_CACHE_CHECK(for sigaction function,
	ac_cv_func_sigaction,
AC_TRY_LINK(
#include <signal.h>
#include <string.h>
, struct sigaction	sigact;

	memset (&sigact, 0, sizeof (sigact));
	/* Verify type signature of handler function. */
	sigact.sa_handler = (RETSIGTYPE (*) (int)) NULL;
	/* Verify that we have SIG_IGN. */
	sigact.sa_handler = SIG_IGN;
	/* Verify that we have SIG_DFL. */
	sigact.sa_handler = SIG_DFL;
	sigaction (SIGTERM, &sigact, NULL);

, ac_cv_func_sigaction=yes, ac_cv_func_sigaction=no))

if test $ac_cv_func_sigaction = yes; then
	AC_DEFINE(HAVE_SIGACTION)
fi

# Override the cached CPLEX settings if environment vars are set!
if test -n "$CPLEX_HEADER"
then
	if test ! -r "$CPLEX_HEADER"
	then
		AC_MSG_ERROR(File given by \$CPLEX_HEADER unreadable!)
	fi
	ac_cv_cplex_header="$CPLEX_HEADER"
fi
if test -n "$CPLEX_LIB"
then
	if test ! -r "$CPLEX_LIB"
	then
		AC_MSG_ERROR(File given by \$CPLEX_LIB unreadable!)
	fi
	ac_cv_cplex_lib="$CPLEX_LIB"
fi

dnl See if the user has specified --with-cplex=no to override.
AC_ARG_WITH(cplex,
[	--with-cplex=no         Do not use CPLEX, even if available],
if test "$withval" = no
then
	with_cplex=no
	# Expunge cplex from the cache!
	ac_cv_cplex_header=''
	ac_cv_cplex_lib=''
else
	if test "$withval" != yes
	then
		echo "Arg to --with-cplex must be yes or no!" 1>&2
		exit 1
	fi
	with_cplex=yes
fi
)

if test "$with_cplex" != no
then
	# Look for the cplex command in the path.  The header and library are
	# often in the same directory as the executable.
	AC_PATH_PROG(ac_cv_path_cplex, cplex)
	cplex_dir='.'
	if test -n "$ac_cv_path_cplex"
	then
		[cplex_dir="`echo $ac_cv_path_cplex | sed 's%/[^/]*$%%'`"]
	fi

	cpx_hdr_dirs="$cplex_dir /usr/include /usr/local/include"
	cpx_lib_dirs="$cplex_dir /usr/lib /usr/ccs/lib /usr/local/lib"

	# Look for cplex.h
	AC_CACHE_CHECK(for cplex.h, ac_cv_cplex_header,
	  AC_FIND_FILE(ac_cv_cplex_header, cplex.h, -r, $cpx_hdr_dirs)
	)

	# Look for libcplex.a
	AC_CACHE_CHECK(for libcplex.a, ac_cv_cplex_lib,
	  AC_FIND_FILE(ac_cv_cplex_lib, libcplex.a, -r, $cpx_lib_dirs)
	)

	if test -z "$ac_cv_cplex_header" || test -z "$ac_cv_cplex_lib"
	then
		# Start looking for CPLEX version 3 stuff...
		# Look for cpxdefs.inc
		AC_CACHE_CHECK(for cpxdefs.inc, ac_cv_cplex_header,
		  AC_FIND_FILE(ac_cv_cplex_header, cpxdefs.inc, -r,
				$cpx_hdr_dirs)
		)

		# Look for cplex.a
		AC_CACHE_CHECK(for cplex.a, ac_cv_cplex_lib,
		  AC_FIND_FILE(ac_cv_cplex_lib, cplex.a, -r,
				$cpx_lib_dirs)
		)
	fi

	# We assume at this point, the variables ac_cv_cplex_header and
	# ac_cv_cplex_lib have been set to the absolute pathnames of the
	# appropriate files, if they are available.
	#
	# The next thing to do is determine the version number of the
	# cplex being referenced, putting it in ac_cv_cplex_version.
	# Note: we use the version number in the header and assume that
	# the library matches!

dnl A macro to determine the CPLEX version number from the cplex header
dnl file...
define(AC_DETECT_CPLEX_VERSION,
[	cpxh="$ac_cv_cplex_header"
	hdrname="`echo $cpxh | sed 's@^.*/@@'`"
	if test "$hdrname" = "cpxdefs.inc"
	then
		ac_cv_cplex_version=30
	else
		vers="`grep Version <$cpxh | head -1 | \
			sed 's@^.*Version[[	 ]]*@@' | \
			sed 's@\([[0-9]][[0-9]]*\)\.\([[0-9]]\).*[$]@\1\2@'`"
		ac_cv_cplex_version="$vers"
	fi
])

	if test -n "$ac_cv_cplex_header" && \
	   test -r "$ac_cv_cplex_header" && \
	   test -n "$ac_cv_cplex_lib"
	then
	  # We have both a header and a library.  Now we need the version.
	  AC_MSG_CHECKING([cplex version number])
	  if test -n "$CPLEX_HEADER" || test -n "$CPLEX_LIB"
	  then
		# User overrode, don't believe the cached value!
		AC_DETECT_CPLEX_VERSION
	  else
		AC_CACHE_VAL(ac_cv_cplex_version, [AC_DETECT_CPLEX_VERSION])
	  fi
	  AC_MSG_RESULT($ac_cv_cplex_version)
	fi
fi

CTYPE_C=''

# Decide (finally) which LP solver to use!

if test "$with_cplex" != no && \
   test -n "$ac_cv_cplex_header" && \
   test -r "$ac_cv_cplex_header" && \
   test -n "$ac_cv_cplex_lib" && \
   test -r "$ac_cv_cplex_lib" && \
   test -n "$ac_cv_cplex_version"
then
	# We have decided to use cplex!
	[cpxhdrdir="`echo $ac_cv_cplex_header | sed 's%/[^/]*$%%'`"]
	[cpxlibdir="`echo $ac_cv_cplex_lib | sed 's%/[^/]*$%%'`"]

	LP_PKG=cplex
	CPLEX_HEADER_DIR="$cpxhdrdir"
	CPLEX_LIB_DIR="$cpxlibdir"
	LP_CFLAGS='-I$(CPLEX_HEADER_DIR)'
	LP_DEPS='$(CPLEX_HEADER_DIR)/cplex.h'
	LP_LIBS='$(CPLEX_LIB_DIR)/libcplex.a'
	AC_DEFINE_UNQUOTED(CPLEX, $ac_cv_cplex_version)
	AC_DEFINE_UNQUOTED(CPLEX_VERSION_STRING, "$ac_cv_cplex_version")

	# Check if this cplex provides CPXcreateprob.
	AC_CPLEX_CHECK_CREATEPROB
	if test "$ac_cv_cplex_has_createprob" = yes
	then
		AC_DEFINE(CPLEX_HAS_CREATEPROB)
	fi

	# Check if we need to use -lpthread and/or ctype.c workaround.
	AC_CPLEX_CHECK_LIBPTHREAD_AND_CTYPE
	if test "$ac_cv_cplex_libpthread" = yes
	then
		LP_LIBS="${LP_LIBS} -lpthread"
	fi
	CLIENT_LP_LIBS="${LP_LIBS}"
	if test "$ac_cv_cplex_needs_ctype" = yes
	then
		# We need to use the "ctype.c" workaround in order to
		# run this old cplex with newer glibc's.
		CTYPE_C="ctype.c"
		AC_DEFINE(NEED_CTYPE_C)
	fi
else
	# We are unable to use cplex.  Use lp_solve instead.
	if (test -n "$ac_cv_cplex_header" || \
	    test -n "$ac_cv_cplex_lib" || \
	    test -n "$ac_cv_prog_cplex") && \
	   test "$with_cplex" != no
	then
		# Warn user that we are falling back to lp_solve!
		AC_MSG_WARN(CPLEX not completely found -- using lp_solve instead...)
	fi
	LP_PKG=lp_solve
	LP_CFLAGS='-I$(LP_SOLVE_DIR)'
	LP_DEPS='$(LP_SOLVE_DIR)/lpkit.h'
	LP_LIBS='$(LP_SOLVE_DIR)/libLPS.a'
	CLIENT_LP_LIBS=''
	AC_DEFINE(LPSOLVE)
fi

AC_SUBST(CPLEX_HEADER_DIR)
AC_SUBST(CPLEX_LIB_DIR)
AC_SUBST(LP_PKG)
AC_SUBST(LP_DEPS)
AC_SUBST(LP_LIBS)
AC_SUBST(LP_CFLAGS)

AC_SUBST(CLIENT_LP_LIBS)

AC_SUBST(CTYPE_C)

dnl Permit the user to specify --with-machine=string to set the
dnl machine description string.  This #define's MACHDESC to
dnl be a C string constant in config.h.
define_machdesc=no
AC_ARG_WITH(machine,
[	--with-machine=string   Force machine description string],
 ac_cv_machine_description_string="$withval"
 define_machdesc=yes,
[if test -n "$ac_cv_machine_description_string"
 then
   define_machdesc=yes
 fi
])

if test $define_machdesc = yes
then
  AC_DEFINE_UNQUOTED(MACHDESC, "${ac_cv_machine_description_string}")
fi

dnl Permit the user to specify environment variables that give
dnl the location of the GNU GMP header and library files.
dnl If the user specifies --with-gmp=no, then ignore these environment
dnl vars.

# Override the cached GMP settings if environment vars are set!
if test "${GMP_HEADER:+set}" = set
then
	if test ! -r "$GMP_HEADER"
	then
		AC_MSG_ERROR(File given by \$GMP_HEADER unreadable!)
	fi
	ac_cv_gmp_header="$GMP_HEADER"
fi
if test "${GMP_LIB:+set}" = set
then
	if test ! -r "$GMP_LIB"
	then
		AC_MSG_ERROR(File given by \$GMP_LIB unreadable!)
	fi
	ac_cv_gmp_lib="$GMP_LIB"
fi

dnl See if the user has specified --with-gmp=no to override.
AC_ARG_WITH(gmp,
[	--with-gmp=no         Do not use GNU GMP, even if available],
if test "$withval" = no
then
	with_gmp=no
	# Expunge GMP from the cache!
	ac_cv_gmp_header=''
	ac_cv_gmp_lib=''
else
	if test "$withval" != yes
	then
		echo "Arg to --with-gmp must be yes or no!" 1>&2
		exit 1
	fi
	with_gmp=yes
fi
)

if test "$with_gmp" != no
then
    if test -n "$ac_cv_gmp_header" && test -n "$ac_cv_gmp_lib"
    then
	GMP_INCLUDE_DIR="`echo $ac_cv_gmp_header | sed -e 's@[[^/]]*$@@' -e 's@/$@@'`"
	if test -z "$GMP_INCLUDE_DIR"; then GMP_INCLUDE_DIR='.'; fi
	GMP_CFLAGS='-I$(GMP_INCLUDE_DIR)'
	GMP_LIBS="$ac_cv_gmp_lib"
	AC_DEFINE(HAVE_GMP)
    else
	AC_GMP_WORKS_NORMALLY(gmp_works=yes, gmp_works=no)
	if test "$gmp_works" = yes
	then
		GMP_INCLUDE_DIR=''
		GMP_CFLAGS=''
		GMP_LIBS='-lgmp'
		AC_DEFINE(HAVE_GMP)
	fi
    fi
fi

AC_SUBST(GMP_INCLUDE_DIR)
AC_SUBST(GMP_CFLAGS)
AC_SUBST(GMP_LIBS)

dnl See if the user has specified --with-triangle=yes to override.
ac_use_triangle=no
AC_ARG_WITH(triangle,
[	--with-triangle[=no]    Use triangle package.],
if test -z "$withval"
then
	ac_use_triangle=no
elif test "$withval" = no
then
	ac_use_triangle=no
elif test "$withval" = yes
then
	ac_use_triangle=yes
else
	echo "Arg to --with-triangle must be yes or no!" 1>&2
	exit 1
fi
)
if test "$ac_use_triangle" = no
then
	TRIANGLE_C=''
else
	TRIANGLE_C=triangle.c
	AC_DEFINE(USE_TRIANGLE)
fi
AC_SUBST(TRIANGLE_C)

dnl these two lines are added by AC_OUTPUT, but I need them to
dnl expand the prefix and exec_prefix variables
test "x$prefix" = xNONE && prefix=$ac_default_prefix
test "x$exec_prefix" = xNONE && exec_prefix='${prefix}'

dnl C macro variable expansion
AC_DEFINE_EXPAND_VALUE(INSTALLDIR_PREFIX, $prefix)
AC_DEFINE_EXPAND_VALUE(INSTALLDIR_EXEC_PREFIX, $exec_prefix)
AC_DEFINE_EXPAND_VALUE(INSTALLDIR_BINDIR, $bindir)
AC_DEFINE_EXPAND_VALUE(INSTALLDIR_DATADIR, $datadir)
AC_DEFINE_EXPAND_VALUE(INSTALLDIR_LIBDIR, $libdir)
AC_DEFINE_EXPAND_VALUE(INSTALLDIR_LIBEXECDIR, $libexecdir)

AC_OUTPUT(Makefile lp_solve_2.3/Makefile,
	echo timestamp >stamp-config-h
)
