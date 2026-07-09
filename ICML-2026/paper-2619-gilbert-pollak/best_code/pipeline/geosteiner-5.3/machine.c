/***********************************************************************

	$Id: machine.c,v 1.13 2022/11/19 13:45:53 warme Exp $

	File:	machine.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1999, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	This file contains routines for obtaining a string that
	describes the machine running the program.

************************************************************************

	Modification Log:

	a-1:	01/12/99	warme
		: Created.
	b-1:	08/05/2002	benny
		: Some small changes to make it easier to avoid memory
		: leaks.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganized include files, apply prefixes.

************************************************************************/

#include "machine.h"

#include "config.h"
#include "logic.h"
#include "memory.h"
#include "steiner.h"
#include <stdio.h>
#include <string.h>
#include "utils.h"


/* Determine how to implement _gst_get_machine_string... */
#if defined(MACHDESC)
	/* This method overrides all other options... */
	#define PRINT_CONSTANT_STRING
#elif defined(UNAME_FUNCTION_WORKS)
	#define USE_UNAME_SYSCALL
#elif defined(HAVE_POPEN) AND defined(HAVE_PCLOSE) AND defined(UNAME_PATH)
	#define USE_UNAME_COMMAND
#else
	#define MACHDESC		"Unknown"
	#define PRINT_CONSTANT_STRING
#endif

#ifdef USE_UNAME_SYSCALL
#include <sys/utsname.h>
#endif


/*
 * Global Routines
 */

char *		_gst_get_machine_string (void);


/*
 * This routine obtains a machine description string from the "uname"
 * system call.
 */

#ifdef USE_UNAME_SYSCALL

	char *
_gst_get_machine_string (void)

{
size_t		len;
char *		s;
struct utsname	un;

	if (uname (&un) < 0) {
		return (_gst_strdup ("Unknown"));
	}

	len	= strlen (un.sysname)
		+ strlen (un.nodename)
		+ strlen (un.release)
		+ strlen (un.version)
		+ strlen (un.machine);

	s = NEWA (len + 5, char);

	sprintf (s, "%s %s %s %s %s",
		 un.sysname, un.nodename, un.release, un.version, un.machine);

	return (s);
}

#endif

/*
 * This routine obtains a machine description string by invoking the
 * "uname -a" command.
 */

#ifdef USE_UNAME_COMMAND

	char *
_gst_get_machine_string (void)

{
int		c;
char *		p;
char *		endp;
size_t		size;
FILE *		fp;
char		command [1024];
char		buffer [80];

	sprintf (command, "%s -a", UNAME_PATH);
	fp = popen (command, "r");
	if (fp EQ NULL) goto dont_know;

	p = buffer;
	endp = p + sizeof (buffer) - 1;
	for (;;) {
		c = getc (fp);
		if (c < 0) break;
		if (p >= endp) break;
		if (c EQ '\n') break;
		*p++ = c;
	}
	*p++ = '\0';
	size = p - buffer;
	p = NEWA (size, char);
	strcpy (p, buffer);

	/* Drain the stream... */
	while (c >= 0) {
		c = getc (fp);
	}

	if (pclose (fp) NE 0) {
dont_know:
		p = _gst_strdup ("Unknown");
	}
	return (p);
}

#endif

/*
 * This routine returns a fixed machine description string.
 */

#ifdef PRINT_CONSTANT_STRING

	char *
_gst_get_machine_string (void)

{
	return (_gst_strdup (MACHDESC));
}

#endif
