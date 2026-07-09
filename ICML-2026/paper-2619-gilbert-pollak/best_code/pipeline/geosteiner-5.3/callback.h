/***********************************************************************

	$Id: callback.h,v 1.2 2022/11/19 13:45:50 warme Exp $

	File:	callback.h
	Rev:	e-1
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Stuff for invoking callback functions, and the various
	Node Accessor API functions.

************************************************************************

	Modification Log:

	e-1:	11/18/2022	warme
		: Created.

************************************************************************/

#ifndef CALLBACK_H
#define	CALLBACK_H

struct bbnode;

extern void	_gst_invoke_bb_callback (int			wherefrom,
					 struct bbnode *	node);

#endif
