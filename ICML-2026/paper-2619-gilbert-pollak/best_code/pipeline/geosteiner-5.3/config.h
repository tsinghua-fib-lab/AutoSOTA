/* config.h.  Generated from config.h.in by configure.  */
/**********************************************************************

	$Id: config.h.in,v 1.18 2022/11/19 13:45:50 warme Exp $

	File:	config.h.in
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1998, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

***********************************************************************

	Compile time switches that are automatically set to the
	proper values by the configuration script.

***********************************************************************

	Modification Log:

	a-1:	12/20/98	warme
		: Created.
	a-2:	02/28/2001	warme
		: Changes for 3.1 release.  Support for GMP,
		:  Intel floating point precision fix, and
		:  stderr being an lvalue.
	c-1:	08/05/2002	benny
		: Improved version support.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
		: Renamed several config switches, removed others.
		: Added CVS Id line.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Make features unconditional.
		: Remove floating-point precision fix option.
	e-4:	11/18/2022	warme
		: Convert 2 features from compile-time options
		:  into algorithm parameters.

***********************************************************************/

#ifndef _CONFIG_H_
#define	_CONFIG_H_

/* Define to CPLEX version number, if using CPLEX. */
/* #undef CPLEX */
/* #undef CPLEX_VERSION_STRING */

/* Define if using lp_solve instead of cplex. */
#define LPSOLVE 1

/* Define if using Shewchuk's triangle package. */
/* #undef USE_TRIANGLE */

/* Define if have GMP library available. */
/* #undef HAVE_GMP */

/* Define if need to work around older CPLEX referencing old <ctype.h> */
/* stuff that newer glibc's do not define. */
/* #undef NEED_CTYPE_C */

/* Define if the CPLEX we are using provides the CPXcreateprob() function. */
/* #undef CPLEX_HAS_CREATEPROB */

/* Define if stderr is an lvalue that can be stored into. */
#define HAVE_STDERR_IS_LVALUE 1

/* Define if times(), struct tms, and CLK_TCK work. */
#define UNIX_CPU_TIME 1

/* Define if uname(), <utsname.h> and struct utsname all work. */
#define UNAME_FUNCTION_WORKS 1

/* Define as a string that describes the machine running this software. */
/* This overrides the use of uname(2) or uname(1) to get such a string. */
/* #undef MACHDESC */

/* Define this if the fsync() function is available */
#define HAVE_FSYNC 1

/* Define this if the link() function is available */
#define HAVE_LINK 1

/* Define this if the rename() function is available */
#define HAVE_RENAME 1

/* Define this if the sync() function is available */
#define HAVE_SYNC 1

/* Define this if the unlink() function is available */
#define HAVE_UNLINK 1

/* Define this if popen is available */
#define HAVE_POPEN 1

/* Define this is pclose is available */
#define HAVE_PCLOSE 1

/* Define as the pathname of the uname command, if available */
#define UNAME_PATH "/usr/bin/uname"

/* Define to empty if the keyword does not work.  */
/* #undef const */

/* Define as __inline if that's what the C compiler calls it.  */
/* #undef inline */

/* Define as void or int, which ever is the return type of signal handlers. */
#define RETSIGTYPE void

/* Define if we have sigaction() and "struct sigaction". */
#define HAVE_SIGACTION 1

/* Define the directories where package is installed */
#define INSTALLDIR_PREFIX "/usr/local"
#define INSTALLDIR_EXEC_PREFIX "/usr/local"
#define INSTALLDIR_BINDIR "/usr/local/bin"
#define INSTALLDIR_DATADIR "/usr/local/share"

/* The current version of the GeoSteiner library */
#define GEOLIB_VERSION_MAJOR 5
#define GEOLIB_VERSION_MINOR 3
#define GEOLIB_VERSION_PATCH 0
#define GEOLIB_VERSION_STRING "5.3.0"

#endif
