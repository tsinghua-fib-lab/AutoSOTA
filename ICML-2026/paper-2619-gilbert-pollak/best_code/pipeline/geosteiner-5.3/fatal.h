/***********************************************************************

	$Id: fatal.h,v 1.4 2022/11/19 13:45:51 warme Exp $

	File:	fatal.h
	Rev:	e-2
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Stuff to report fatal software errors.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Split off from general.h.
		: Added FATAL_ERROR and FATAL_ERROR_IF().
	e-2:	09/24/2016	warme
		: Apply prefixes.  Remove obsolete FATAL() macro.

************************************************************************/

#ifndef FATAL_H
#define FATAL_H

#include <stddef.h>

extern void	_gst_fatal (const char *	filename,
			    int			lineno);

#define FATAL_ERROR						\
	do {							\
		_gst_fatal (__FILE__, __LINE__);		\
	} while (0)

#define FATAL_ERROR_IF(condition)				\
	do {							\
		if (condition) {				\
			_gst_fatal (__FILE__, __LINE__);	\
		}						\
	} while (0)

#endif	/* FATAL_H */
