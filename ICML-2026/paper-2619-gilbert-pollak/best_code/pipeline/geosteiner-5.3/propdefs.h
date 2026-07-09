/***********************************************************************

	$Id: propdefs.h,v 1.6 2022/11/19 13:45:54 warme Exp $

	File:	propdefs.h
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	12/28/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

************************************************************************/

#ifndef PROPDEFS_H
#define PROPDEFS_H

/* Define all properties here. */
/* Columns are symbol and value. */
#define HG_PROPS(f) \
 f(HALF_FST_COUNT,		10000) \
 f(GENERATION_TIME,		20000) \
 f(MST_LENGTH,			20001) \
 f(PRUNING_TIME,		20002) \
 f(INTEGRALITY_DELTA,		20003) \
 f(NAME,			30000) \
	/* end of list */

#define SOLVER_PROPS(f) \
 f(ROOT_OPTIMAL,		11000) \
 f(ROOT_LPS,			11001) \
 f(NUM_NODES,			11002) \
 f(NUM_LPS,			11003) \
 f(INIT_PROWS,			11004) \
 f(INIT_PNZ,			11005) \
 f(INIT_LPROWS,			11006) \
 f(INIT_LPNZ,			11007) \
 f(ROOT_PROWS,			11008) \
 f(ROOT_PNZ,			11009) \
 f(ROOT_LPROWS,			11010) \
 f(ROOT_LPNZ,			11011) \
 f(FINAL_PROWS,			11012) \
 f(FINAL_PNZ,			11013) \
 f(FINAL_LPROWS,		11014) \
 f(FINAL_LPNZ,			11015) \
 f(LOWER_BOUND,			11016) \
 f(CPU_TIME,			21000) \
 f(ROOT_TIME,			21001) \
 f(ROOT_LENGTH,			21002) \
	/* end of list */

#endif
