/***********************************************************************

	$Id: memory.h,v 1.14 2022/11/19 13:45:53 warme Exp $

	File:	memory.h
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

	a-1:	05/31/02	warme
		: Created.  Split off from general.h.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef MEMORY_H
#define MEMORY_H

#define _GNU_SOURCE

#include <stddef.h>

/*
 * Optionally keep an eye on the memory...
 */

#ifdef WATCH_MEMORY

 #ifdef __GNUC__

  extern void *	       _gst_allocate_memory (size_t, char *, char *, int);
  extern void	       _gst_free_memory (void *, char *, char *, int);

  #define _gst_new(a)	_gst_allocate_memory (a, __FILE__, __FUNCTION__, __LINE__)
  #undef free
  #define free(a)	_gst_free_memory (a, __FILE__, __FUNCTION__, __LINE__)

 #else

  extern void *		_gst_allocate_memory (size_t, char *, int);
  extern void		_gst_free_memory (void *, char *, int);

  #define _gst_new(a)	_gst_allocate_memory (a, __FILE__, __LINE__)
  #undef free
  #define free(a)	_gst_free_memory (a, __FILE__, __LINE__)

 #endif
 extern void		_gst_check_memory ();
 #define CHECK_MEMORY	_gst_check_memory ();

#else

 #define CHECK_MEMORY
 extern void *		_gst_new (size_t);

#endif

/*
 * Useful macros.
 */

#define	NEW(type)	((type *) _gst_new (sizeof (type)))
#define	NEWA(n, type)	((type *) _gst_new ((size_t) ((n) * sizeof (type))))

#endif	/* MEMORY_H */
