/***********************************************************************

	$Id: parmdefs.h,v 1.35 2023/04/03 17:21:13 warme Exp $

	File:	parmdefs.h
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Added missing include file.
		: Add new save_format and save_int_numbits.
	e-3:	09/24/2016	warme
		: Make features unconditional.
	e-4:	11/18/2022	warme
		: Added several new parameters.
	e-5:	04/03/2023	warme
		: Added local_cuts_trace_style.

************************************************************************/

#ifndef PARMDEFS_H
#define PARMDEFS_H

/* Define the default value of the MAX_CUTSET_ENUMERATE_COMPS parameter. */
/* If the number of connected components is this number N or fewer, we	 */
/* will generate a "cutset" constraint (or two subtours) for each of the */
/* 2**N-2 combinations of connected components.  Otherwise, we only	 */
/* generate one "cutset" constraint per component.  We should experiment */
/* a bit with this parameter, but I bet that N=2 is probably best...	 */

#define MCEC	5	/* Bash only 5 CComps (was 11 in version 3.1). */

/* Define all of the INTEGER parameters right here. */
/* Columns are symbol, value, variable, min, max, and default values. */

#define INTPARMS(f) \
 f(GROUP_DEFINITION,		1000, group_definition,		 0, 1, 0) \
 f(EPS_MULT_FACTOR,		1001, eps_mult_factor,		 1, INT_MAX, 32) \
 f(INITIAL_EQPOINTS_TERMINAL,	1002, initial_eqpoints_terminal, 1, INT_MAX, 100) \
 f(MAX_FST_SIZE,		1003, max_fst_size,		 2, INT_MAX, INT_MAX) \
 f(MULTIPLE_PRECISION,		1004, multiple_precision,	 0, 2, 0) \
 f(EFST_HEURISTIC,		1005, efst_heuristic,		 0, 1, 0) \
 f(TARGET_POOL_NON_ZEROS,	1006, target_pool_non_zeros,	 0, INT_MAX, 0) \
 f(LP_SOLVE_PERTURB,		1007, lp_solve_perturb,		 0, 1, 0) \
 f(LP_SOLVE_SCALE,		1008, lp_solve_scale,		 0, 1, 0) \
 f(CPLEX_MIN_ROWS,		1009, cplex_min_rows,		 0, INT_MAX, 0) \
 f(CPLEX_MIN_NZS,		1010, cplex_min_nzs,		 0, INT_MAX, 0) \
 f(BRANCH_VAR_POLICY,		1011, branch_var_policy,	 0, 3, 1) \
 f(CHECK_BRANCH_VARS_THOROUGHLY,1012, check_branch_vars_thoroughly, 1, 1000, 1) \
 f(CHECK_ROOT_CONSTRAINTS,	1013, check_root_constraints,	 0, 1, 0) \
 f(LOCAL_CUTS_MODE,		1014, local_cuts_mode,		 0, 3, 0) \
 f(LOCAL_CUTS_MAX_VERTICES,	1015, local_cuts_max_vertices,	 0, INT_MAX, 80) \
 f(LOCAL_CUTS_MAX_EDGES,	1016, local_cuts_max_edges,	 0, INT_MAX, 256) \
 f(LOCAL_CUTS_MAX_DEPTH,	1017, local_cuts_max_depth,	 -1, 2, 1) \
 f(LOCAL_CUTS_TRACE_DEPTH,	1018, local_cuts_trace_depth,	 -1, 2, 0) \
 f(SEED_POOL_WITH_2SECS,	1019, seed_pool_with_2secs,	 0, 1, 1) \
 f(NUM_FEASIBLE_SOLUTIONS,	1020, num_feasible_solutions,	 1, INT_MAX, 1) \
 f(MAX_FEASIBLE_UPDATES,	1021, max_feasible_updates,	 0, INT_MAX, 0) \
 f(SOLVER_ALGORITHM,		1022, solver_algorithm,		 0, 2, 0) \
 f(INCLUDE_CORNERS,		1023, include_corners,		 0, 1, 0) \
 f(BACKTRACK_MAX_VERTS,		1024, backtrack_max_verts,	 0, 32, 8) \
 f(BACKTRACK_MAX_EDGES,		1025, backtrack_max_edges,	 0, 32, 12) \
 f(MAX_BACKTRACKS,		1026, max_backtracks,		 0, INT_MAX, 10000) \
 f(SAVE_FORMAT,			1027, save_format,		 0, 4, 3) \
 f(GRID_OVERLAY,		1028, grid_overlay,		 0, 1, 1) \
 f(BSD_METHOD,			1029, bsd_method,		 0, 2, 0) \
 f(MAX_CUTSET_ENUMERATE_COMPS,	1030, max_cutset_enumerate_comps,0, 11, MCEC) \
 f(SEC_ENUM_LIMIT,		1031, sec_enum_limit,		 0, 16, 10) \
 f(SAVE_INT_NUMBITS,		1032, save_int_numbits,		32, INT_MAX, 64) \
 f(SPARSE_SUBTOURS,		1033, sparse_subtours,		 0, 1, 1) \
 f(ZERO_WEIGHT_CUTSETS_METHOD,	1034, zero_weight_cutsets_method,0, 1, 0) \
 f(BB_NODE_LIMIT,		1035, bb_node_limit,		 0, INT_MAX, 0) \
 f(BB_LP_LIMIT,			1036, bb_lp_limit,		 0, INT_MAX, 0) \
 f(STRENGTHEN_REDUCE,		1037, strengthen_reduce,	 0, 1, 1) \
 f(GENERATE_UNSTRENGTHENED,	1038, generate_unstrengthened,	 0, 1, 1) \
 f(INITIAL_PRIMAL_HEURISTIC,	1039, initial_primal_heuristic,	 0, 1, 0) \
 f(INITIAL_PRIMAL_HEUR_STOP,	1040, initial_primal_heur_stop,	 0, 1, 0) \
 f(LOCALCUTS_TRACE_STYLE,	1041, local_cuts_trace_style,	 0, 1, 0) \
	/* end of list */

/* Define all of the DOUBLE parameters right here. */
/* Columns are symbol, value, variable, min, max, and default values. */

#define DBLPARMS(f) \
 f(INITIAL_UPPER_BOUND,		2000, initial_upper_bound,	  -DBL_MAX, DBL_MAX, DBL_MAX) \
 f(LOCAL_CUTS_VERTEX_THRESHOLD,	2001, local_cuts_vertex_threshold,0, 1, 0.75) \
 f(CPU_TIME_LIMIT,		2002, cpu_time_limit,		  0, DBL_MAX, 0) \
 f(GAP_TARGET,			2003, gap_target,		  1, DBL_MAX, 1) \
 f(UPPER_BOUND_TARGET,		2004, upper_bound_target,	  -DBL_MAX, DBL_MAX, -DBL_MAX) \
 f(LOWER_BOUND_TARGET,		2005, lower_bound_target,	  -DBL_MAX, DBL_MAX, DBL_MAX) \
 f(CHECKPOINT_INTERVAL,		2006, checkpoint_interval, 0, 1000000.0, 3600) \
	/* end of list */

/* Define all of the STRING parameters right here. */
/* Columns are symbol, value, variable, validation func, and default value. */

#define STRPARMS(f) \
  f(CHECKPOINT_FILENAME,	3000, checkpoint_filename,	NULL, NULL) \
  f(MERGE_CONSTRAINT_FILES,	3001, merge_constraint_files,	NULL, NULL) \
	/* end of list */

/* Define all of the CHANNEL parameters right here. */
/* Columns are symbol, value, variable, validation func, and default value. */

#define CHNPARMS(f) \
 f(DETAILED_TIMINGS_CHANNEL,	4000, detailed_timings_channel, NULL) \
 f(PRINT_SOLVE_TRACE,		4001, print_solve_trace, NULL) \
	/* end of list */

#endif /* PARMDEFS_H */
