/***********************************************************************

	$Id: logic.h,v 1.8 2022/11/19 13:45:53 warme Exp $

	File:	logic.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	05/31/02	warme
		: Created.  Split off from general.h.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Move NEW() and NEWA() into memory.h.

************************************************************************/

#ifndef	LOGIC_H
#define	LOGIC_H

/*
 * Macros to protect us from some of C's sharpest edges.
 */

#define	NOT	!
#define	AND	&&
#define	OR	||
#define	FALSE	0
#define	TRUE	1
#define	EQ	==
#define	NE	!=

#endif	/* LOGIC_H */
