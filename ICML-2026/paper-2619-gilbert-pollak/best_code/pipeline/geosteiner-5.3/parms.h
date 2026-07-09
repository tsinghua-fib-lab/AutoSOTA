/***********************************************************************

	$Id: parms.h,v 1.2 2022/11/19 13:45:53 warme Exp $

	File:	parms.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	The parameter system.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef PARMS_H
#define	PARMS_H

struct environment;
struct gst_parmdefs;

extern void	_gst_initialize_parameters (struct environment * env);
extern void	_gst_shutdown_parameters (struct gst_parmdefs * pdefs);

#endif
