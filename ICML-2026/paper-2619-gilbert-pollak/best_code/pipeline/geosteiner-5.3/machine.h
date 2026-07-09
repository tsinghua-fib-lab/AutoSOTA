/***********************************************************************

	$Id: machine.h,v 1.2 2022/11/19 13:45:53 warme Exp $

	File:	machine.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Routine to get a description of the machine that is
	running the program.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef MACHINE_H
#define	MACHINE_H

extern char *	_gst_get_machine_string (void);

#endif
