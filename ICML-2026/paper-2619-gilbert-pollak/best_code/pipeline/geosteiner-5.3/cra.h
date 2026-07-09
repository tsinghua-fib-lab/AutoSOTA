/***********************************************************************

	$Id: cra.h,v 1.8 2022/11/19 13:45:50 warme Exp $

	File:	cra.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Closest Rational Approximation routine.

************************************************************************

	Modification Log:

	a-1:	07/28/2000	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Apply prefixes.

************************************************************************/


#ifndef CRA_H
#define	CRA_H

extern double	_gst_cra (double	z,
			  double *	numout,
			  double *	denout);

#endif
