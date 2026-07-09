/***********************************************************************

	$Id: memory.c,v 1.22 2022/11/19 13:45:53 warme Exp $

	File:	memory.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	When WATCH_MEMORY is defined then the functions in this file will
	replace new/free. When allocating, a front and back wall is added to
	the allocation and when freeing these walls are checked. At the end of
	a program CHECK_MEMORY can be used to verify that all of the
	allocated memory has been freed (no memory leaks).

************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created. Derived from code by Allan Odgaard.
	b-1:	04/24/2014	warme
		: Fix 64-bit architecture issues by providing a
		:  portable way to format size_t and 16-bit offset
		:  values with fprintf().
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize header files, apply prefixes.
		: Fix -Wall issues.

************************************************************************/

#include "memory.h"

#include "gsttypes.h"
#include "logic.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stddef.h>
#undef free

/*
 * Global Routines
 */

void *		_gst_allocate_memory (size_t, char *,
#ifdef __GNUC__
				      char *,
#endif
				      int);
void		_gst_free_memory (void *, char *,
#ifdef __GNUC__
				  char *,
#endif
				  int);
void		_gst_check_memory ();

/*
 * Local Routines
 */

static void	mem_dump (const int8u *, size_t, const int8u *);
static void	cv_size_t_to_decimal (char * buf, size_t size);

/*
 * Local Values
 */

static long int TotalMemUsage = 0;
static long int Chunks = 0;

static const int8u pad[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xC0, 0xDE, 0xDB, 0xAD };

/*
 * Local Types
 */

struct memory_chunk {
	struct memory_chunk *	next;
	struct memory_chunk *	prev;
	char *			filename;   /* in which file was it allocated */
#if __GNUC__
	char *			function_name; /* in which function */
#endif
	int			line_number;   /* at which line */
	size_t			size;	       /* how large is the allocation */
	int8u			front[16 * sizeof (pad)];	/* front wall */
	int8u			back[16 * sizeof (pad)];	/* back wall */
};

struct memory_chunk *first = NULL;

/*
 * Dump a piece of memory to stderr.
 */
	static
	void
mem_dump (

const int8u *		mem,	/* IN - memory pointer */
size_t			size,	/* IN - size of memory chunk */
const int8u *		base	/* IN - end of memory chunk */
)
{
const int8u *adr = mem;
int j = 0;
size_t i;
char hex[40], asc[18];
int offset;

	hex[0] = '\0';
	/* for (i = 0; i < size; i++, j++) */
	for (i = 0; i < (size > 256 ? 256 : size); i++, j++) {
		if (j EQ 16) {
			offset = (adr - base) & 0xFFFF;
			fprintf (stderr, " %04X: %-35.35s \"%-16.16s\"\n",
				 offset, hex, asc);
			j = 0;
			hex[0] = '\0';
			adr = &mem[i];
		}
		else if ((j > 0) AND ((j % 4) EQ 0)) {
			strcat (hex, " ");
		}

		sprintf (&hex[strlen (hex)], "%02X", mem[i]);
		asc[j] = isprint (mem[i]) ? mem[i] : ' ';
	}

	if (j > 0) {
		asc[j] = '\0';
		offset = (adr - base) & 0xFFFF;
		fprintf (stderr, " %04X: %-35.35s \"%-16.16s\"\n",
			 offset, hex, asc);
	}
}

/*
 * Allocate a piece of memory.
 */
	void *
_gst_allocate_memory (

size_t		size,		/* IN - size of memory to be allocated */
char *		filename,	/* IN - */
#ifdef __GNUC__
char *		funcname,	/* IN - */
#endif
int		line		/* IN - */
)
{
struct memory_chunk *	m;
int8u *			res;
unsigned int		i;

	m = (struct memory_chunk *)
		malloc (sizeof (struct memory_chunk) + size);

	if (m EQ NULL) {
		fprintf (stderr, "Allocation failed\n");
		abort ();
	}
#if 0
	else {
		char	szbuf [32];
		cv_size_t_to_decimal (szbuf, size);
		fprintf (stderr, "Allocated %s bytes at %p\n", szbuf, m);
	}
#endif

	m -> next = first;
	m -> prev = NULL;
	m -> size = size;
	m -> filename = filename;
#if __GNUC__
	m -> function_name = funcname;
#endif
	m -> line_number = line;

	if (first) {
		first -> prev = m;
	}
	first = m;

	res = &m -> back[0];
	for (i = 0; i < 16 * sizeof (pad); i++) {
		m -> front[i] = res[size+i] = pad[i % sizeof (pad)];
	}

	memset (res, 0x7F, size);
	TotalMemUsage += size;
	Chunks++;
	return res;
}

/*
 * Free a piece of memory. Check that the borders of the chunk have not been
 * overwritten and that the chunk exists in the set of allocated memory chunks.
 */

	void
_gst_free_memory (

void *		r,
char *		filename,
#ifdef __GNUC__
char *		funcname,
#endif
int		line
)
{
struct memory_chunk *	mem;
unsigned int		i;
size_t			size;
int8u *			res;
int			front;
int			back;

	if (r EQ NULL) {
		return;
	}

	mem = (struct memory_chunk *)(((char *)r) - offsetof (struct memory_chunk, back));

#if 0
	{
		/* this checks if the memory chunk is valid		*/
		/* -- but it's a linear check (switch to a better	*/
		/* datastructure)					*/
		struct memory_chunk * known = first;
		while ((known NE NULL) AND (known NE mem)) {
			known = known -> next;
		}

		if (known EQ NULL) {
			char	szbuf [32];
			cv_size_t_to_decimal (szbuf, mem -> size);
			fprintf (stderr,
				 "*** freeing unknown memory chunk: %p (%s bytes)!\n",
				 r, szbuf);
			return;
		}
	}
#endif

	if (mem -> prev NE NULL) {
		mem -> prev -> next = mem -> next;
	}
	else {
		first = mem -> next;
	}

	if (mem -> next NE NULL) {
		mem -> next -> prev = mem -> prev;
	}

	size = mem -> size;

	res = &mem -> back[0];
	front = FALSE;
	back = FALSE;
	for (i = 0; i < 16 * sizeof (pad); i++) {
		if (mem -> front[i] NE pad[i % sizeof (pad)]) {
			front = TRUE;
		}

		if (res[size+i] NE pad[i % sizeof (pad)]) {
			back = TRUE;
		}
	}

	if (front) {
		char	szbuf [32];
		cv_size_t_to_decimal (szbuf, size);
		fprintf (stderr,
			 "Freeing %s bytes at %p (%s: %d"
#ifdef __GNUC__
			 " - %s"
#endif
			 ")\n",
			 szbuf, mem, filename, line
#ifdef __GNUC__
			 , funcname
#endif
			);
		fprintf (stderr, "\nFront wall of %s bytes broken!\n", szbuf);
		mem_dump (&mem -> front[0], 16 * sizeof (pad), res);
		mem_dump (res, size, res);
	}

	if (back) {
		if (NOT front) {
			char	szbuf [32];
			cv_size_t_to_decimal (szbuf, size);
			fprintf (stderr,
				 "Freeing %s bytes at %p (%s: %d"
#ifdef __GNUC__
				 " - %s"
#endif
				 ")\n",
				 szbuf, mem, filename, line
#ifdef __GNUC__
				 , funcname
#endif
				);
			fprintf (stderr, "\nBack wall of %s bytes broken\n",
				 szbuf);
			mem_dump (res, size, res);
		}
		mem_dump (&res[size], 16 * sizeof (pad), res);
	}

	--Chunks;
	TotalMemUsage -= size;
	memset (r, 0xF7, size);
#if 0
	{
		char	szbuf [32];
		cv_size_t_to_decimal (szbuf, size);
		fprintf (stderr,
			 "Freeing %s bytes at %p (%s: %d"
#ifdef __GNUC__
			 " - %s"
#endif
			 ")\n",
			 szbuf, mem, filename, line
#ifdef __GNUC__
			 , funcname
#endif
			 );
	}
#endif
	free (mem);
}

/*
 * Check that all chunks have been freed. Dump any unfreed chunks.
 */

	void
_gst_check_memory ()

{
struct memory_chunk *	mem;

	if (TotalMemUsage NE 0) {
		mem = first;
		fprintf (stderr, "Memory leak at %ld byte(s) (%ld chunk(s))\n", TotalMemUsage, Chunks);
		while (mem NE NULL) {
			char	szbuf [32];
			cv_size_t_to_decimal (szbuf, mem -> size);
			fprintf (stderr, "\n%4s byte(s) not freed\n", szbuf);
			fprintf (stderr,
				" File: %s,"
#if __GNUC__
				" Function: %s,"
#endif
				" Line number: %d\n",
				mem -> filename,
#if __GNUC__
				mem -> function_name,
#endif
				mem -> line_number);
			mem_dump (&mem -> back[0], mem -> size, &mem -> back[0]);
			mem = mem -> next;
		}
	}
}

/*
 * Convert the given size_t value to a decimal ASCII string.
 */

	static
	void
cv_size_t_to_decimal (

char *		buf,		/* IN/OUT: buffer to receive ASCII string */
size_t		size		/* IN: size_t value to convert */
)
{
int		dig;
char		tmp;
char *		p1;
char *		p2;

	if (size EQ 0) {
		strcpy (buf, "0");
		return;
	}

	p1 = buf;
	while (size > 0) {
		dig = (size % 10);
		*p1++ = '0' + dig;
		size /= 10;
	}
	*p1 = '\0';

	/* Reverse the digit sequence. */
	--p1;
	p2 = buf;
	while (p2 < p1) {
		tmp = *p2;
		*p2 = *p1;
		*p1 = tmp;
		++p2;
		--p1;
	}
}
