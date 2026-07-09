/***********************************************************************

	$Id: errordefs.h,v 1.19 2023/04/03 17:41:54 warme Exp $

	File:	errordefs.h
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	07/11/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	11/18/2022	warme
		: Added INVALID_SOLVER.
	e-4:	04/03/2023	warme
		: Added BACKTRACK_BAD_COSTS.

************************************************************************/

#ifndef ERRORDEFS_H
#define ERRORDEFS_H

/* Define all error values here. */
/* Columns are symbol and value. */
#define ERRORVALS(f) \
 f(UNDEFINED,				1000) \
 f(LIBRARY_CLOSED,			1001) \
 f(PROPERTY_NOT_FOUND,			1002) \
 f(PROPERTY_TYPE_MISMATCH,		1003) \
 f(BACKTRACK_OVERFLOW,			1004) \
 f(SOLUTION_NOT_AVAILABLE,		1005) \
 f(RANK_OUT_OF_RANGE,			1006) \
 f(INVALID_METRIC,			1007) \
 f(NO_EMBEDDING,			1008) \
 f(ALREADY_CLOSED,			1009) \
 f(LP_SOLVER_ACTIVE,			1010) \
 f(LOAD_ERROR,				1011) \
 f(INVALID_NUMBER_OF_TERMINALS,		1012) \
 f(PARAMETER_VALUE_OUT_OF_RANGE,	1013) \
 f(UNKNOWN_PARAMETER_ID,		1014) \
 f(INVALID_PROPERTY_LIST,		1015) \
 f(INVALID_HYPERGRAPH,			1016) \
 f(INVALID_NUMBER_OF_VERTICES,		1017) \
 f(INVALID_NUMBER_OF_EDGES,		1018) \
 f(INVALID_EDGE,			1019) \
 f(INVALID_VERTEX,			1020) \
 f(INVALID_DIMENSION,			1021) \
 f(NO_STEINERS_ALLOWED,			1022) \
 f(INVALID_CHANNEL,			1023) \
 f(INVALID_CHANNEL_OPTIONS,		1024) \
 f(INVALID_PARAMETERS_OBJECT,		1025) \
 f(INVALID_PARAMETER_TYPE,		1026) \
 f(EFST_GENERATOR_DISABLED,		1029) \
 f(RFST_GENERATOR_DISABLED,		1030) \
 f(UFST_GENERATOR_DISABLED,		1031) \
 f(FST_PRUNER_DISABLED,			1032) \
 f(INVALID_SOLVER,			1033) \
 f(BACKTRACK_BAD_COSTS,			1034) \
	/* end of list */
/* Error values */

#endif
