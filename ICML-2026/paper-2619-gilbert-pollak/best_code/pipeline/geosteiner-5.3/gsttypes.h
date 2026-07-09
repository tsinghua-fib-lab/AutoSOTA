/***********************************************************************

	$Id: gsttypes.h,v 1.2 2022/11/19 13:45:52 warme Exp $

	File:	gsttypes.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Primitive types.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from general.h.

************************************************************************/

#ifndef GSTTYPES_H
#define	GSTTYPES_H

/*
 * Typedef's to insulate us from the ugliness of C types.
 */

typedef unsigned char		int8u;
#if	defined(__STDC__) && ((__STDC__+0)==1)
typedef signed char		int8s;
#else
typedef char			int8s;
#endif
typedef unsigned short		int16u;
typedef short			int16s;
typedef unsigned int		int32u;
typedef int			int32s;
typedef unsigned long		int64u;
typedef char			bool;

#endif
