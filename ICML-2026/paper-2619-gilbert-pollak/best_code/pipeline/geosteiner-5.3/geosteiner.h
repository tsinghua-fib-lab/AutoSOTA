/***********************************************************************

 GeoSteiner 5.3.0 header file

 Copyright (c) 2004, 2022 by David M. Warme, Pawel Winter & Martin Zachariasen.
 All rights reserved.

************************************************************************/

#ifndef GEOSTEINER_H
#define GEOSTEINER_H

#include <stdio.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
 #define _GST_PRINTF_ARGS(a,b) __attribute__ ((__format__ (__printf__,a,b)))
#else
 #define _GST_PRINTF_ARGS(a,b)
#endif

/* Hypergraph properties */

#define GST_PROP_HG_HALF_FST_COUNT                        10000
#define GST_PROP_HG_GENERATION_TIME                       20000
#define GST_PROP_HG_MST_LENGTH                            20001
#define GST_PROP_HG_PRUNING_TIME                          20002
#define GST_PROP_HG_INTEGRALITY_DELTA                     20003
#define GST_PROP_HG_NAME                                  30000

/* Solver properties */

#define GST_PROP_SOLVER_ROOT_OPTIMAL                      11000
#define GST_PROP_SOLVER_ROOT_LPS                          11001
#define GST_PROP_SOLVER_NUM_NODES                         11002
#define GST_PROP_SOLVER_NUM_LPS                           11003
#define GST_PROP_SOLVER_INIT_PROWS                        11004
#define GST_PROP_SOLVER_INIT_PNZ                          11005
#define GST_PROP_SOLVER_INIT_LPROWS                       11006
#define GST_PROP_SOLVER_INIT_LPNZ                         11007
#define GST_PROP_SOLVER_ROOT_PROWS                        11008
#define GST_PROP_SOLVER_ROOT_PNZ                          11009
#define GST_PROP_SOLVER_ROOT_LPROWS                       11010
#define GST_PROP_SOLVER_ROOT_LPNZ                         11011
#define GST_PROP_SOLVER_FINAL_PROWS                       11012
#define GST_PROP_SOLVER_FINAL_PNZ                         11013
#define GST_PROP_SOLVER_FINAL_LPROWS                      11014
#define GST_PROP_SOLVER_FINAL_LPNZ                        11015
#define GST_PROP_SOLVER_LOWER_BOUND                       11016
#define GST_PROP_SOLVER_CPU_TIME                          21000
#define GST_PROP_SOLVER_ROOT_TIME                         21001
#define GST_PROP_SOLVER_ROOT_LENGTH                       21002

/* Error codes */

#define GST_ERR_UNDEFINED                                 1000
#define GST_ERR_LIBRARY_CLOSED                            1001
#define GST_ERR_PROPERTY_NOT_FOUND                        1002
#define GST_ERR_PROPERTY_TYPE_MISMATCH                    1003
#define GST_ERR_BACKTRACK_OVERFLOW                        1004
#define GST_ERR_SOLUTION_NOT_AVAILABLE                    1005
#define GST_ERR_RANK_OUT_OF_RANGE                         1006
#define GST_ERR_INVALID_METRIC                            1007
#define GST_ERR_NO_EMBEDDING                              1008
#define GST_ERR_ALREADY_CLOSED                            1009
#define GST_ERR_LP_SOLVER_ACTIVE                          1010
#define GST_ERR_LOAD_ERROR                                1011
#define GST_ERR_INVALID_NUMBER_OF_TERMINALS               1012
#define GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE              1013
#define GST_ERR_UNKNOWN_PARAMETER_ID                      1014
#define GST_ERR_INVALID_PROPERTY_LIST                     1015
#define GST_ERR_INVALID_HYPERGRAPH                        1016
#define GST_ERR_INVALID_NUMBER_OF_VERTICES                1017
#define GST_ERR_INVALID_NUMBER_OF_EDGES                   1018
#define GST_ERR_INVALID_EDGE                              1019
#define GST_ERR_INVALID_VERTEX                            1020
#define GST_ERR_INVALID_DIMENSION                         1021
#define GST_ERR_NO_STEINERS_ALLOWED                       1022
#define GST_ERR_INVALID_CHANNEL                           1023
#define GST_ERR_INVALID_CHANNEL_OPTIONS                   1024
#define GST_ERR_INVALID_PARAMETERS_OBJECT                 1025
#define GST_ERR_INVALID_PARAMETER_TYPE                    1026
#define GST_ERR_EFST_GENERATOR_DISABLED                   1029
#define GST_ERR_RFST_GENERATOR_DISABLED                   1030
#define GST_ERR_UFST_GENERATOR_DISABLED                   1031
#define GST_ERR_FST_PRUNER_DISABLED                       1032
#define GST_ERR_INVALID_SOLVER                            1033
#define GST_ERR_BACKTRACK_BAD_COSTS                       1034

/* Parameters */

#define GST_PARAM_GROUP_DEFINITION                        1000
#define GST_PARAM_EPS_MULT_FACTOR                         1001
#define GST_PARAM_INITIAL_EQPOINTS_TERMINAL               1002
#define GST_PARAM_MAX_FST_SIZE                            1003
#define GST_PARAM_MULTIPLE_PRECISION                      1004
#define GST_PARAM_EFST_HEURISTIC                          1005
#define GST_PARAM_TARGET_POOL_NON_ZEROS                   1006
#define GST_PARAM_LP_SOLVE_PERTURB                        1007
#define GST_PARAM_LP_SOLVE_SCALE                          1008
#define GST_PARAM_CPLEX_MIN_ROWS                          1009
#define GST_PARAM_CPLEX_MIN_NZS                           1010
#define GST_PARAM_BRANCH_VAR_POLICY                       1011
#define GST_PARAM_CHECK_BRANCH_VARS_THOROUGHLY            1012
#define GST_PARAM_CHECK_ROOT_CONSTRAINTS                  1013
#define GST_PARAM_LOCAL_CUTS_MODE                         1014
#define GST_PARAM_LOCAL_CUTS_MAX_VERTICES                 1015
#define GST_PARAM_LOCAL_CUTS_MAX_EDGES                    1016
#define GST_PARAM_LOCAL_CUTS_MAX_DEPTH                    1017
#define GST_PARAM_LOCAL_CUTS_TRACE_DEPTH                  1018
#define GST_PARAM_SEED_POOL_WITH_2SECS                    1019
#define GST_PARAM_NUM_FEASIBLE_SOLUTIONS                  1020
#define GST_PARAM_MAX_FEASIBLE_UPDATES                    1021
#define GST_PARAM_SOLVER_ALGORITHM                        1022
#define GST_PARAM_INCLUDE_CORNERS                         1023
#define GST_PARAM_BACKTRACK_MAX_VERTS                     1024
#define GST_PARAM_BACKTRACK_MAX_EDGES                     1025
#define GST_PARAM_MAX_BACKTRACKS                          1026
#define GST_PARAM_SAVE_FORMAT                             1027
#define GST_PARAM_GRID_OVERLAY                            1028
#define GST_PARAM_BSD_METHOD                              1029
#define GST_PARAM_MAX_CUTSET_ENUMERATE_COMPS              1030
#define GST_PARAM_SEC_ENUM_LIMIT                          1031
#define GST_PARAM_SAVE_INT_NUMBITS                        1032
#define GST_PARAM_SPARSE_SUBTOURS                         1033
#define GST_PARAM_ZERO_WEIGHT_CUTSETS_METHOD              1034
#define GST_PARAM_BB_NODE_LIMIT                           1035
#define GST_PARAM_BB_LP_LIMIT                             1036
#define GST_PARAM_STRENGTHEN_REDUCE                       1037
#define GST_PARAM_GENERATE_UNSTRENGTHENED                 1038
#define GST_PARAM_INITIAL_PRIMAL_HEURISTIC                1039
#define GST_PARAM_INITIAL_PRIMAL_HEUR_STOP                1040
#define GST_PARAM_LOCALCUTS_TRACE_STYLE                   1041
#define GST_PARAM_INITIAL_UPPER_BOUND                     2000
#define GST_PARAM_LOCAL_CUTS_VERTEX_THRESHOLD             2001
#define GST_PARAM_CPU_TIME_LIMIT                          2002
#define GST_PARAM_GAP_TARGET                              2003
#define GST_PARAM_UPPER_BOUND_TARGET                      2004
#define GST_PARAM_LOWER_BOUND_TARGET                      2005
#define GST_PARAM_CHECKPOINT_INTERVAL                     2006
#define GST_PARAM_CHECKPOINT_FILENAME                     3000
#define GST_PARAM_MERGE_CONSTRAINT_FILES                  3001
#define GST_PARAM_DETAILED_TIMINGS_CHANNEL                4000
#define GST_PARAM_PRINT_SOLVE_TRACE                       4001

/* Parameter values */

#if 0
/* For GST_PARAM_GROUP_DEFINITION */
#define GST_PVAL_GROUP_DEFINITION_ATLEAST               0
#define GST_PVAL_GROUP_DEFINITION_EXACTLY               1
#endif

/* For GST_PARAM_INCLUDE_CORNERS */
#define GST_PVAL_INCLUDE_CORNERS_DISABLE                0
#define GST_PVAL_INCLUDE_CORNERS_ENABLE                 1

/* For GST_PARAM_EFST_HEURISTIC */
#define GST_PVAL_EFST_HEURISTIC_SLL                     0
#define GST_PVAL_EFST_HEURISTIC_ZW                      1

/* For GST_PARAM_MULTIPLE_PRECISION */
#define GST_PVAL_MULTIPLE_PRECISION_OFF                 0
#define GST_PVAL_MULTIPLE_PRECISION_ONE_ITER            1
#define GST_PVAL_MULTIPLE_PRECISION_MORE_ITER           2

/* For GST_PARAM_BSD_METHOD */
#define GST_PVAL_BSD_METHOD_DYNAMIC                     0
#define GST_PVAL_BSD_METHOD_CONSTANT                    1
#define GST_PVAL_BSD_METHOD_LOGARITHMIC                 2

/* For GST_PARAM_LP_SOLVE_PERTURB */
#define GST_PVAL_LP_SOLVE_PERTURB_DISABLE               0
#define GST_PVAL_LP_SOLVE_PERTURB_ENABLE                1

/* For GST_PARAM_LP_SOLVE_SCALE */
#define GST_PVAL_LP_SOLVE_SCALE_DISABLE                 0
#define GST_PVAL_LP_SOLVE_SCALE_ENABLE                  1

/* For GST_PARAM_SOLVER_ALGORITHM */
#define GST_PVAL_SOLVER_ALGORITHM_AUTO                  0
#define GST_PVAL_SOLVER_ALGORITHM_BRANCH_AND_CUT        1
#define GST_PVAL_SOLVER_ALGORITHM_BACKTRACK_SEARCH      2

/* For GST_PARAM_BRANCH_VAR_POLICY */
#define GST_PVAL_BRANCH_VAR_POLICY_NAIVE                0
#define GST_PVAL_BRANCH_VAR_POLICY_SMART                1
#define GST_PVAL_BRANCH_VAR_POLICY_PROD                 2
#define GST_PVAL_BRANCH_VAR_POLICY_WEAK                 3

/* For GST_PARAM_SEED_POOL_WITH_2SECS */
#define GST_PVAL_SEED_POOL_WITH_2SECS_DISABLE           0
#define GST_PVAL_SEED_POOL_WITH_2SECS_ENSABLE           1

/* For GST_PARAM_CHECK_ROOT_CONSTRAINTS */
#define GST_PVAL_CHECK_ROOT_CONSTRAINTS_DISABLE         0
#define GST_PVAL_CHECK_ROOT_CONSTRAINTS_ENABLE          1

/* For GST_PARAM_LOCAL_CUTS_MODE */
#define GST_PVAL_LOCAL_CUTS_MODE_DISABLE                0
#define GST_PVAL_LOCAL_CUTS_MODE_SUBTOUR_RELAXATION     1
#define GST_PVAL_LOCAL_CUTS_MODE_SUBTOUR_COMPONENTS     2
#define GST_PVAL_LOCAL_CUTS_MODE_BOTH                   3

/* For GST_PARAM_LOCAL_CUTS_MAX_DEPTH */
#define GST_PVAL_LOCAL_CUTS_MAX_DEPTH_DISABLE           0
#define GST_PVAL_LOCAL_CUTS_MAX_DEPTH_ONELEVEL          1
#define GST_PVAL_LOCAL_CUTS_MAX_DEPTH_TWOLEVELS         2
#define GST_PVAL_LOCAL_CUTS_MAX_DEPTH_ANYLEVEL          -1

/* For GST_PARAM_LOCAL_CUTS_TRACE_DEPTH */
#define GST_PVAL_LOCAL_CUTS_TRACE_DEPTH_DISABLE         0
#define GST_PVAL_LOCAL_CUTS_TRACE_DEPTH_ONELEVEL        1
#define GST_PVAL_LOCAL_CUTS_TRACE_DEPTH_TWOLEVELS       2
#define GST_PVAL_LOCAL_CUTS_TRACE_DEPTH_ANYLEVEL        -1

/* For GST_PARAM_LOCAL_CUTS_TRACE_STYLE */
#define GST_PVAL_LOCAL_CUTS_TRACE_STYLE_NORMAL          0
#define GST_PVAL_LOCAL_CUTS_TRACE_STYLE_VERBOSE         1

/* For GST_PARAM_SPARSE_SUBTOURS */
#define GST_PVAL_SPARSE_SUBTOURS_DISABLE		0
#define GST_PVAL_SPARSE_SUBTOURS_ENABLE			1

/* For GST_PARAM_ZERO_WEIGHT_CUTSETS_METHOD */
#define GST_PVAL_ZERO_WEIGHT_CUTSETS_METHOD_SUBTOURS	0
#define GST_PVAL_ZERO_WEIGHT_CUTSETS_METHOD_CUTSET	1

/* For GST_PARAM_STRENGTHEN_REDUCE */
#define GST_PVAL_STRENGTHEN_REDUCE_DISABLE		0
#define GST_PVAL_STRENGTHEN_REDUCE_ENABLE		1

/* For GST_PARAM_GENERATE_UNSTRENGTHENED */
#define GST_PVAL_GENERATE_UNSTRENGTHENED_DISABLE	0
#define GST_PVAL_GENERATE_UNSTRENGTHENED_ENABLE		1

/* For GST_PARAM_INITIAL_PRIMAL_HEURISTIC */
#define GST_PVAL_INITIAL_PRIMAL_HEURISTIC_DISABLE	0
#define GST_PVAL_INITIAL_PRIMAL_HEURISTIC_ENABLE	1

/* For GST_PARAM_INITIAL_PRIMAL_HEUR_STOP */
#define GST_PVAL_INITIAL_PRIMAL_HEUR_STOP_DISABLE	0
#define GST_PVAL_INITIAL_PRIMAL_HEUR_STOP_ENABLE	1

/* For GST_PARAM_SAVE_FORMAT */
#define GST_PVAL_SAVE_FORMAT_ORLIBRARY                  0
#define GST_PVAL_SAVE_FORMAT_STEINLIB                   1
#define GST_PVAL_SAVE_FORMAT_VERSION2                   2
#define GST_PVAL_SAVE_FORMAT_VERSION3                   3
#define GST_PVAL_SAVE_FORMAT_STEINLIB_INT               4

/* For GST_PARAM_GRID_OVERLAY */
#define GST_PVAL_GRID_OVERLAY_DISABLE                   0
#define GST_PVAL_GRID_OVERLAY_ENABLE                    1

/* Solution status codes */

#define GST_STATUS_OPTIMAL     0  /* Optimal solution is available */
#define GST_STATUS_INFEASIBLE  1  /* Problem is infeasible */
#define GST_STATUS_FEASIBLE    2  /* Search incomplete, feasible known */
#define GST_STATUS_NO_FEASIBLE 3  /* Search incomplete, no feasible known */
#define GST_STATUS_NO_SOLUTION 4  /* Solver never invoked, or problem changed */

/* Black-box pointer types */

typedef struct gst_channel *    gst_channel_ptr;
typedef struct gst_hypergraph * gst_hg_ptr;
typedef struct gst_metric *     gst_metric_ptr;
typedef struct gst_node *	gst_node_ptr;
typedef struct gst_param *      gst_param_ptr;
typedef struct gst_proplist *   gst_proplist_ptr;
typedef struct gst_scale_info * gst_scale_info_ptr;
typedef struct gst_solver *     gst_solver_ptr;

struct cpxenv;

/****************************************************************/

/*
 * Opening and closing \geosteiner{} environment
 * 
 * _functions
 * The environment encapsulates licensing information and 
 * platform-specific data. If CPLEX is used as LP solver, the CPLEX
 * environment is stored in the environment. 
 * 
 * The environment is a single global variable. No explicit user
 * references to the environment are possible, but the environment 
 *   must be initialized by calling the gst_open_geosteiner()
 * function before any other library functions can be invoked.
 * 
 * In the reminder of this section, we present each of the functions in
 * the library related to the environment.
 */

/****************************************/

/*
 * gst_open_geosteiner
 * 
 * can be in two major states open or closed.  The
 * initial state is always closed.  This routine transitions
 * from the closed state to the open state by 
 * initializing the environment.
 * No other library function may be called when is 
 * closed.  In a multi-threaded environment, it is the
 * application's responsibility to ensure that no calls to other
 * library functions are either pending or initiated until
 * is in the open state --- which begins as soon as
 * this routine returns with a status code of zero.
 * 
 * Note that the function does not open the LP solver
 * (e.g., CPLEX). This is done automatically the first time the LP solver
 * environment is accessed; however, it can also be done explicitly
 * using the gst_open_lpsolver() function.
 * An existing CPLEX environment can also be attached to the 
 * environment.  See gst_attach_cplex(); this is only relevant
 * for CPLEX versions of the library.
 */

int gst_open_geosteiner (void);

/*
 * Returns status code (which is zero if was successfully opened).
 */

/****************************************/

/*
 * gst_close_geosteiner
 * 
 * Transition from the open to the closed state.
 * Conceptually, enters the closed state the
 * very instant this routine is called. 
 * In a multi-threaded environment, it is the application's
 * responsibility to ensure that no calls to other library
 * functions are pending at the time this routine is invoked. 
 */

int gst_close_geosteiner (void);

/*
 * Returns error code (which is zero if was successfully closed).
 */

/****************************************/

/*
 * gst_version_string
 * 
 * Return version number as a character string.
 */

const char * gst_version_string (void);

/*
 * Returns null-terminated string giving the version number.
 */

/****************************************/

/*
 * gst_version
 * 
 * Return version number as an integer with the following
 * decimal interpretation: XXXYYYZZZ, where XXX is the major version, YYY
 * is the minor version and ZZZ is the patch-level.
 */

int gst_version (void);

/*
 * Returns integer representing the version number.
 */

/****************************************/

/*
 * gst_open_lpsolver
 * 
 * Initialize LP solver (e.g., CPLEX) environment. It is not necessary to
 * open the LP solver explicitly, since this is done automatically the
 * first time the LP solver is needed. However, it might be advantageous to
 * ensure that the LP solver has been successfully opened and is available
 * for use before starting a long run.
 */

int gst_open_lpsolver (void);

/*
 * Returns value zero if the LP solver was opened successfully or already
 * was open.
 */

/****************************************/

/*
 * gst_close_lpsolver
 * 
 * Close LP solver environment. In the case where the LP solver was
 * attached, e.g., using gst_attach_cplex(), then this
 * routine detaches but does not close the LP solver.
 */

int gst_close_lpsolver (void);

/*
 * Returns value zero if the solver was closed successfully or already
 * was closed. 
 */

/****************************************/

/*
 * gst_lpsolver_version_string
 * 
 * Return the name of the configured LP solver and its version number as a
 * string.
 */

const char* gst_lpsolver_version_string (void);

/*
 * Returns zero-terminated string giving the LP solver name and version.
 */

/****************************************/

/*
 * gst_attach_cplex
 * 
 * Provided only for CPLEX versions of the library.  Attach an existing
 * CPLEX environment to .  Certain applications may wish to use
 * CPLEX before, during and/or after they use .  This function
 * permits such applications to use an existing CPLEX environment rather
 * than letting attempt to open CPLEX itself (which would
 * fail if CPLEX were already open). A non-NULL CPLEX environment
 * that was attached using gst_attach_cplex() will not be closed
 * when gst_close_geosteiner() is called. 
 */

void gst_attach_cplex (struct cpxenv*  envp);

/*
 * No return value.
 */

/****************************************/

/*
 * gst_detach_cplex
 * 
 * Provided only for CPLEX versions of the library. Detach and return a
 * previously attached CPLEX environment. Does not close the CPLEX
 * environment. 
 */

struct cpxenv* gst_detach_cplex ();

/*
 * Return value is NULL if no CPLEX environment is currently attached.
 * 
 * An example is given with the documentation of 
 * gst_attach_cplex() on page~_attach_cplex.
 */

/****************************************************************/

/*
 * High-level optimization functions
 * 
 * _level_functions
 * The high-level functions give the user easy access to the basic
 * algorithms in the library. There are two types of functions:
 * Firstly, there are functions that solve Steiner tree problems in the
 * plane by passing a set of point coordinates; secondly, the MSTHG
 * problem can be solved by giving a description of the hypergraph
 * instance. 
 * 
 * All functions have a parameter set as argument. This parameter set can
 * be created and modified using the functions described in
 * Section~_functions. However, default parameters are
 * used for all parameters if a NULL pointer is passed as
 * parameter set.
 *  
 */

/****************************************/

/*
 * gst_smt
 * 
 * Given a set of points (or terminals) in the plane, construct an SMT for the
 * points. The metric used for the SMT construction must be specified.
 * (Dedicated functions for specific metrics are given on the following pages.)
 * The length of the constructed SMT, the Steiner points and the list of
 * line segments in the SMT are returned.
 * 
 * Any of the output parameters may be set to NULL if the corresponding
 * output is not needed. It is the responsibility of the user to allocate
 * sufficient memory for the output arrays.
 * *-0.3cm
 */

int gst_smt (int             nterms,
             double*         terms,
             double*         length,
             int*            nsps,
             double*         sps,
             int*            nedges,
             int*            edges,
             int*            status,
             gst_metric_ptr  metric,
             gst_param_ptr   param);

/*
 * Returns value zero if an SMT was computed and non-zero otherwise.
 * See Figure~:demo2 on page~:demo2 or the example
 * file 2.c for an example of how to use gst_smt().
 */

/****************************************/

/*
 * gst_esmt
 * 
 * Given a set of points (or terminals) in the plane, construct an 
 * Euclidean SMT for the points. The length of the constructed SMT, the
 * Steiner points and the list of line segments in the SMT are returned.
 * 
 * Any of the output parameters may be set to NULL if the corresponding
 * output is not needed. It is the responsibility of the user to allocate
 * sufficient memory for the output arrays.
 */

int gst_esmt (int            nterms, 
              double*        terms, 
              double*        length,
              int*           nsps,  
              double*        sps, 
              int*           nedges,
              int*           edges, 
              int*           status,
              gst_param_ptr  param);

/*
 * Returns value zero if an SMT was computed and non-zero otherwise.
 * 
 * An example is given in Section~_level_interfaces.
 */

/****************************************/

/*
 * gst_rsmt
 * 
 * Given a set of points (or terminals) in the plane, construct a 
 * rectilinear SMT for the points. The length of the constructed SMT,
 * the Steiner points and the list of line segments in the SMT are
 * returned.
 * 
 * Any of the output parameters may be set to NULL if the corresponding
 * output is not needed. It is the responsibility of the user to allocate
 * sufficient memory for the output arrays.
 */

int gst_rsmt (int            nterms,
              double*        terms,
              double*        length,
              int*           nsps,
              double*        sps,
              int*           nedges,
              int*           edges,
              int*           status,
              gst_param_ptr  param);

/*
 * Returns value zero if an SMT was computed and non-zero otherwise.
 * 
 * An example is given in Section~_level_interfaces.
 */

/****************************************/

/*
 * gst_osmt
 * 
 * Given a set of points (or terminals) in the plane, construct an 
 * octilinear SMT for the points.  The length of the constructed SMT, the
 * Steiner points and the list of line segments in the SMT are returned.
 * 
 * Any of the output parameters may be set to NULL if the corresponding
 * output is not needed. It is the responsibility of the user to allocate
 * sufficient memory for the output arrays.
 */

int gst_osmt (int            nterms,
              double*        terms,
              double*        length,
              int*           nsps,
              double*        sps,
              int*           nedges,
              int*           edges,
              int*           status,
              gst_param_ptr  param);

/*
 * Returns value zero if an SMT was computed and non-zero otherwise.
 * 
 * An example is given in Section~_level_interfaces.
 */

/****************************************/

/*
 * gst_hgmst
 * 
 * Given an edge-weighted hypergraph, construct a minimum spanning tree
 * (MST) in this hypergraph.
 * 
 * Any of the output parameters may be set to NULL if the corresponding
 * output is not needed. It is the responsibility of the user to allocate
 * sufficient memory for the output arrays.
 */

int gst_hgmst (int            nverts,
               int            nedges,
               int*           edge_sizes,
               int*           edges,
               double*        weights,
               double*        length,
               int*           nmstedges,
               int*           mstedges,
               int*           status,
               gst_param_ptr  param);

/*
 * Returns value zero if an MST was computed and non-zero otherwise.
 * 
 * % Just to move example to next page
 */

/****************************************************************/

/*
 * Parameter setting and querying functions
 * 
 * _functions
 * A parameter set is an object that holds values for all parameters in the
 * library.
 * The library provides the following operations on parameter sets:
 * 
 *   create a parameter set having ``default'' values,
 *   change parameter settings in a parameter set,
 *   query the current, default, minimum and maximum values
 *         of any parameter,
 *   query the type of a parameter,
 *   copy an existing parameter set,
 *   free a parameter set.
 * 
 * Parameter sets have type _param_ptr. Various library
 * functions require a parameter set to be provided as 
 * an argument. In all such cases it is valid for the caller to pass a
 * NULL pointer, in which case default settings will be used for all
 * parameters.
 * 
 * Each supported parameter has a specific type.  When querying the type of
 * a parameter, the library responds with an integer value that denotes the
 * corresponding parameter type.  The parameter types supported, together
 * with the integer values that denote them are as follows:
 * 
 * 
 *  
 *   :parmtypes
 *   
 *    Type		& Macro Name		& Value 
 *    
 *    int			& GST_PARAMTYPE_INTEGER	&	1 
 *    double		& GST_PARAMTYPE_DOUBLE	&	2 
 *    char*		& GST_PARAMTYPE_STRING	&	3 
 *    gst_channel_ptr	& GST_PARAMTYPE_CHANNEL	&	4 
 *   
 *  
 * 
 * 
 * Externally each parameter has a unique number defined by a
 * GST_PARAM macro (see Appendix~). This macro is
 * used as an argument to the parameter get/set functions. Note that
 * there are distinct parameter get/set functions for each parameter
 * type.
 */

#define GST_PARAMTYPE_INTEGER    1
#define GST_PARAMTYPE_DOUBLE     2
#define GST_PARAMTYPE_STRING     3
#define GST_PARAMTYPE_CHANNEL    4

/****************************************/

/*
 * gst_create_param
 * 
 * Create a new parameter set with default parameters.
 */

gst_param_ptr gst_create_param (int*  status);

/*
 * Returns new parameter set with default parameters.
 */

/****************************************/

/*
 * gst_copy_param
 * 
 * Copy all parameter values from one parameter set into another.
 */

int gst_copy_param (gst_param_ptr  dst,
                    gst_param_ptr  src);

/*
 * Returns zero if the parameter set was copied successfully.
 */

/****************************************/

/*
 * gst_free_param
 * 
 * Free parameter set.  Freeing a parameter set that is still referenced
 * by any other object (e.g., by a problem solution state
 * object) produces undefined behavior. 
 */

int gst_free_param (gst_param_ptr  param);

/*
 * Returns zero if the parameter set was freed successfully.
 */

/****************************************/

/*
 * gst_set_dbl_param
 * 
 * Change value of a specified double parameter in a given parameter set.
 */

int gst_set_dbl_param (gst_param_ptr  param,
                       int            whichparam,
                       double         newvalue);

/*
 * Returns zero if the parameter was set successfully.
 */

/****************************************/

/*
 * gst_get_dbl_param
 * 
 * Get current value of a specified double parameter from a given parameter set.
 */

int gst_get_dbl_param (gst_param_ptr  param,
                       int            whichparam,
                       double*        value);

/*
 * Returns zero if the parameter was accessed successfully.
 */

/****************************************/

/*
 * gst_query_dbl_param
 * 
 * Query properties of a specified double parameter in a given parameter set.
 */

int gst_query_dbl_param (gst_param_ptr  param,
                         int            whichparam,
                         double*        current_value,
                         double*        default_value,
                         double*        min_value,
                         double*        max_value);

/*
 * Each of the last four arguments may be NULL if the
 * corresponding value is not needed.
 * 
 * Returns zero if the parameter was queried successfully.
 * 
 */

/****************************************/

/*
 * gst_set_int_param
 * 
 * Change value of a specified integer parameter in a given parameter set.
 */

int gst_set_int_param (gst_param_ptr  param,
                       int            whichparam,
                       int            newvalue);

/*
 * Returns zero if the parameter was set successfully.
 */

/****************************************/

/*
 * gst_get_int_param
 * 
 * Get current value of a specified integer parameter from a given parameter
 * set.
 */

int gst_get_int_param (gst_param_ptr  param,
                       int            whichparam,
                       int*           value);

/*
 * Returns zero if the parameter was accessed successfully.
 */

/****************************************/

/*
 * gst_query_int_param
 * 
 * Query properties of a specified integer parameter in a given parameter
 * set.
 */

int gst_query_int_param (gst_param_ptr  param,
                         int            whichparam,
                         int*           current_value,
                         int*           default_value,
                         int*           min_value,
                         int*           max_value);

/*
 * Each of the last four arguments may be NULL if the corresponding
 * value is not needed.
 * 
 * Returns zero if the parameter was queried successfully.
 * 
 */

/****************************************/

/*
 * gst_set_str_param
 * 
 * Change value of a specified string parameter in a given parameter set.
 */

int gst_set_str_param (gst_param_ptr  param, 
                       int            whichparam, 
                       const char*    str);

/*
 * Returns zero if the parameter was set successfully.
 */

/****************************************/

/*
 * gst_get_str_param
 * 
 * Get current value of a specified string parameter in a given parameter
 * set.
 */

int gst_get_str_param (gst_param_ptr  param, 
                       int            whichparam,
                       int*           length,
                       char*          str);

/*
 * Returns zero if the parameter was accessed successfully.
 * 
 */

/****************************************/

/*
 * gst_set_chn_param
 * 
 * Change value of a specified channel parameter in a given parameter set.
 */

int gst_set_chn_param (gst_param_ptr    param,
                       int              whichparam,
                       gst_channel_ptr  chan);

/*
 * Returns zero if the parameter was set successfully.
 */

/****************************************/

/*
 * gst_get_chn_param
 * 
 * Get current value of a specified channel parameter from a given parameter
 * set.
 */

int gst_get_chn_param (gst_param_ptr     param,
                       int               whichparam,
                       gst_channel_ptr*  chan);

/*
 * Returns zero if the parameter was accessed successfully.
 */

/****************************************/

/*
 * gst_get_param_id
 * 
 * Translate a parameter name into the corresponding parameter id.
 */

int gst_get_param_id (const char*       param_name,
                      int*              param_id);

/*
 * Returns zero if the _name was recognized and the parameter
 * ID was successfully found.
 */

/****************************************/

/*
 * gst_get_param_type
 * 
 * Get the type of a specified parameter id.
 */

int gst_get_param_type (int   whichparam,
                        int*  type);

/*
 * Returns zero if the type was found successfully.
 */

/****************************************/

/*
 * gst_set_param
 * 
 * Set the value of a named parameter from the given string.  This
 * routine permits the value of any integer, double or string parameter
 * to be set to the value given in text string form.  This is a
 * convenient way to set parameters from command line arguments.
 */

int gst_set_param (gst_param_ptr  param,
                   const char*    name,
                   const char*    value);

/*
 */

/****************************************************************/

/*
 * Metric setting and querying functions
 * 
 * _functions
 * 
 * The support of different metrics in the library is primarily
 * handled by metric objects. Some functions in the library use these metric
 * objects automatically, e.g., gst_esmt(), while others require one to
 * specify a metric object, e.g., gst_smt(). The metric objects provide a
 * simple way to make general applications support several different metrics. An
 * example of this can be found in the demo program 2.c which
 * is the code for a small program supporting all metrics supported by .
 * 
 * Two $L_p$-metrics, $L_1$ (rectilinear) and $L_2$ (Euclidean), are
 * supported. Also, all uniform metrics --- so-called
 * $$-metrics --- are supported. The latter are metrics where only
 * a limited number $2$ of equally-spaced orientations are
 * allowed for the edges in a solution. For $= 2$ this is
 * identical to the rectilinear metric, $L_1$. 
 * 
 * When a metric object has been created, the distance between two points
 * in the metric can be obtained by calling gst_distance(). This 
 * is especially useful for the $$-metrics for which efficient
 * calculation is non-trivial.
 * 
 * The following macros are used for identifying the supported metrics:
 * 
 *  
 *   :metrictypes
 *   Metric Type & Macro Name & Value 
 *   
 *   None	&	GST_METRIC_NONE    &	0 
 *   $L_p$	&	GST_METRIC_L	    &	1 
 *   Uniform &	GST_METRIC_UNIFORM &	2 
 *  
 * 
 */

#define GST_METRIC_NONE         0
#define GST_METRIC_L            1
#define GST_METRIC_UNIFORM      2

/****************************************/

/*
 * gst_create_metric
 * 
 * A metric is defined by a type and a parameter. For the $L_p$-metric
 * this parameter $p$ must be either 1 or 2, and for the $$-metric
 * we must have $2$.
 * 
 * Note that even though the $L_1$-metric and the $$-metric
 * with parameter 2 are the same (rectilinear metric), you cannot expect
 * them to give exactly the same results when used to solve Steiner
 * problems. The first one will result in the use of a dedicated FST
 * generator for the rectilinear problem and the latter will result in
 * the use of a general FST generator for $$-metrics. If you are
 * aiming for speed then use the $L_1$-metric. 
 */

gst_metric_ptr gst_create_metric (int   type,
                                  int   parameter,
                                  int*  status);

/*
 * Returns new metric object.
 */

/****************************************/

/*
 * gst_free_metric
 * 
 * Free an existing metric object. Freeing a metric object that is still
 * referenced by any other object (e.g., a hypergraph
 * object) produces undefined behavior.
 */

int gst_free_metric (gst_metric_ptr  metric);

/*
 * Returns zero if operation was successful.
 */

/****************************************/

/*
 * gst_copy_metric
 * 
 * Copy attributes from one metric object to another.
 */

int gst_copy_metric (gst_metric_ptr  dst, 
                     gst_metric_ptr  src);

/*
 * Returns zero if metric object was copied.
 */

/****************************************/

/*
 * gst_distance
 * 
 * Compute the distance between two points under a given metric.
 */

double gst_distance (gst_metric_ptr  metric,
                     double          x1,
                     double          y1,
                     double          x2,
                     double          y2);

/*
 * Returns the distance. Returned value is always zero if metric type is "None".
 */

/****************************************/

/*
 * gst_get_metric_info
 * 
 * Get the information about a metric object.
 */

int gst_get_metric_info (gst_metric_ptr  metric,
                         int*            type,
                         int*            parameter);

/*
 * Returns zero if operation was successful. Either of the last two
 * arguments may be NULL if the corresponding value is not needed.
 */

/****************************************************************/

/*
 * Property list setting and querying functions
 * 
 * _functions
 * Property lists can be used to hold values which are rarely updated (the
 * data structure holding the information cannot be queried/updated in
 * constant time). The following basic operations are provided by the library:
 * 
 *   create an empty property list,
 *   set/create a value in a property list,
 *   delete a value from a property list,
 *   get a value in a property list,
 *   query the type of a property,
 *   copy a property list,
 *   free a property list (including its content).
 * 
 * A property list has type _proplist_ptr and a property is known by
 * its property ID (a macro name which expands to a signed integer).
 * 
 * The main purpose of property lists is to make extra information about the
 * solution process available to the user through a simple interface. Any property
 * ID with a value larger than or equal to zero is reserved by the library.
 * Negative values can be freely used by the user. The property ID values (and
 * their macro names) currently in use can be found in
 * Appendices~_properties and ~_properties.
 * 
 * Note that there are distinct property get/set functions for different
 * property types. The type of a given property --- which is an integer
 * --- can be queried. The supported property types, together with the
 * integer values that denote them are as follows:
 * 
 * 
 *  
 *   :proptypes
 *   
 *    Type		& Macro Name		& Value 
 *    
 *    int			& GST_PROPTYPE_INTEGER	&	1 
 *    double		& GST_PROPTYPE_DOUBLE		&	2 
 *    char*		& GST_PROPTYPE_STRING		&	3 
 *   
 *  
 * 
 */

/****************************************/

/*
 * gst_create_proplist
 * 
 * Create a new empty property list.
 */

gst_proplist_ptr 
    gst_create_proplist (int*  status);

/*
 * Returns new property list.
 */

/****************************************/

/*
 * gst_free_proplist
 * 
 * Free an existing property list.  Freeing a property list that is still
 * referenced by existing GeoSteiner objects (e.g., hypergraphs and
 * solvers) results in undefined behavior. In most cases it is an error to
 * free a property list that was not obtained via a call to
 * gst_create_proplist().
 */

int gst_free_proplist (gst_proplist_ptr  plist);

/*
 * Returns a status code (zero if operation was successful and non-zero otherwise).
 */

/****************************************/

/*
 * gst_copy_proplist
 * 
 * Empty the destination property list and copy all properties into it from
 * the source property list.
 */

int gst_copy_proplist (gst_proplist_ptr  dst,
                       gst_proplist_ptr  src);

/*
 * Returns zero if the property list was copied successfully.
 */

#define GST_PROPTYPE_INTEGER    1
#define GST_PROPTYPE_DOUBLE     2
#define GST_PROPTYPE_STRING     3

/****************************************/

/*
 * gst_get_property_type
 * 
 * Query the type of a given property.
 */

int gst_get_property_type (gst_proplist_ptr  plist,
                           int               propid,
                           int*              type);

/*
 * Return a status code (zero if operation was successful and non-zero otherwise).
 */

/****************************************/

/*
 * gst_delete_property
 * 
 * Remove any value that might be defined for the given property ID,
 * regardless of type.
 */

int gst_delete_property (gst_proplist_ptr   plist,
                         int                propid);

/*
 * Returns zero if the property was successfully deleted from the property
 * list.
 * Returns GST_ERR_INVALID_PROPERTY_LIST if the property list
 * itself is invalid.
 * Returns GST_ERR_PROPERTY_NOT_FOUND if no property having the
 * given ID exists.
 */

/****************************************/

/*
 * gst_get_dbl_property 
 * 
 * Get the value of a specified double property from a given property
 * list.  The specified property must be of type double or an error
 * is returned.
 * ID values greater than or equal to zero are reserved for 's
 * use.
 * Negative ID values can be freely used by user applications.
 */

int gst_get_dbl_property (gst_proplist_ptr  plist, 
                          int               propid,
                          double*           value);

/*
 * Returns zero if the property was accessed successfully.
 * Returns GST_ERR_PROPERTY_NOT_FOUND if no property having the
 * given ID exists.
 * Returns GST_ERR_PROPERTY_TYPE_MISMATCH if the property exists but
 * does not have type double.
 */

/****************************************/

/*
 * gst_get_int_property 
 * 
 * Get the value of a specified property from the given property list.
 * The specified property must be of type integer or an error is
 * returned.
 * ID values greater than or equal to zero are reserved for 's
 * use.
 * Negative ID values can be freely used by user applications.
 */

int gst_get_int_property (gst_proplist_ptr  plist, 
                          int               propid,
                          int*              value);

/*
 * Returns zero if the property was accessed successfully.
 * Returns GST_ERR_PROPERTY_NOT_FOUND if no property having the
 * given ID exists.
 * Returns GST_ERR_PROPERTY_TYPE_MISMATCH if the property exists but
 * does not have type integer.
 */

/****************************************/

/*
 * gst_get_str_property 
 * 
 * Get the value of a specified property from the given property list.
 * The specified property must be of type string or an error is
 * returned.
 * ID values greater than or equal to zero are reserved for 's
 * use.
 * Negative ID values can be freely used by user applications.
 */

int gst_get_str_property (gst_proplist_ptr  plist,
                          int               propid,
                          int*              length,
                          char*             str);

/*
 * Returns zero if the property was accessed successfully.
 * Returns GST_ERR_PROPERTY_NOT_FOUND if no property having the
 * given ID exists.
 * Returns GST_ERR_PROPERTY_TYPE_MISMATCH if the property exists but
 * does not have type string.
 * 
 */

/****************************************/

/*
 * gst_get_properties
 * 
 * Retrieve all property IDs and their types from the given property list.
 */

int gst_get_properties (gst_proplist_ptr  plist,
                        int*              count,
                        int*              propids,
                        int*              types);

/*
 * Returns zero if the properties were successfully retrieved.
 */

/****************************************/

/*
 * gst_set_dbl_property
 * 
 * Change or create a specified property in the given property list.
 * The property is added to the list if not already present. If the
 * property already exists, its type is forced to be double.
 * It is legal to do this with any property list.
 */

int gst_set_dbl_property (gst_proplist_ptr  plist, 
                          int               propid, 
                          double            value);

/*
 * Returns zero if the property was set successfully.
 */

/****************************************/

/*
 * gst_set_int_property
 * 
 * Change or create a a specified property in the given property list.
 * The property is added to the list if not already present. If the
 * property already exists, its type is forced to be integer.
 * It is legal to do this with any property list.
 */

int gst_set_int_property (gst_proplist_ptr  plist, 
                          int               propid, 
                          int               value);

/*
 * Returns zero if the property was set successfully.
 */

/****************************************/

/*
 * gst_set_str_property
 * 
 * Change or create a specified property in the given property list.
 * The property is added to the list if not already present. If the
 * property already exists, its type is forced to be string.
 * It is legal to do this with any property list.
 */

int gst_set_str_property (gst_proplist_ptr  plist, 
                          int               propid, 
                          const char*       value);

/*
 * Returns zero if the property was set successfully.
 */

/****************************************************************/

/*
 * Hypergraph functions
 * 
 * _functions
 * 
 * The hypergraph object represents an arbitrary hypergraph that can be
 * decorated with a variety of additional (and optional) data.  For
 * example, the edges can be given weights.  In general, the goal of
 * is to find a spanning tree of minimum total weight using the
 * edges of the hypergraph.
 * 
 * In this section we document all of the operations provided for creating,
 * destroying and manipulating hypergraph objects.
 * 
 * Hypergraphs can be embedded in the plane: Vertices can be given
 * coordinates and hyperedges can be associated with trees in the plane. 
 * Also, every hypergraph has an associated metric object
 * (Section~_functions), a scaling object
 * (Section~_functions) and a property list
 * (Section~_functions). 
 * 
 * The library interfaces have been designed to permit maximum flexibility
 * in using the various operations provided.  For example, it is
 * intended that the user be able to define a hypergraph, solve it, modify
 * some attributes of the hypergraph (e.g., change some of the edge costs),
 * and re-solve the modified problem.  The library should be 
 * smart enough to know when the problem can be re-solved starting from the
 * most recent solution state --- and when it is necessary to discard the
 * previous solution state and re-solve the current problem from scratch.
 */

/****************************************/

/*
 * gst_create_hg
 * 
 * Create an instance of an empty hypergraph.  The hypergraph initially has
 * no vertices and no edges.  After creating an empty hypergraph, the next
 * step is normally to give it the desired number of vertices
 * using gst_set_hg_number_of_vertices(), and then add the
 * edges using gst_set_hg_edges(). Doing the steps in this order
 * avoids the failure that would result from attempting to add edges that
 * refer to non-existent vertices.
 */

gst_hg_ptr gst_create_hg (int*  status);

/*
 * Returns new hypergraph object.
 */

/****************************************/

/*
 * gst_copy_hg
 * 
 * Make a copy of a given hypergraph. Any data associated with the
 * destination hypergraph is discarded, and the following attributes are
 * copied from the source hypergraph (if present): vertices, edges, edge
 * weights, metric object info, scale object info, property list, vertex
 * embedding, and edge embedding.
 */

int gst_copy_hg (gst_hg_ptr  dst,
                 gst_hg_ptr  src);

/*
 * Returns zero if the hypergraph was copied successfully.
 */

/****************************************/

/*
 * gst_copy_hg_edges
 * 
 * Make a copy of a given hypergraph with a subset of the original edges.
 * Any data associated with the destination hypergraph is
 * discarded, and the following attributes are copied from the source
 * hypergraph (if present): vertices, (subset of) edges, (subset of) edge
 * weights, metric object info, scale object info, property list, vertex 
 * embedding, and edge embedding. 
 */

int gst_copy_hg_edges (gst_hg_ptr  dst, 
                       gst_hg_ptr  src, 
                       int         nedges, 
                       int*        edges);

/*
 * Returns zero if (a subset of) the hypergraph was copied successfully.
 */

/****************************************/

/*
 * gst_free_hg
 * 
 * Remove a hypergraph and free all associated memory, including associated
 * properties. 
 */

int gst_free_hg (gst_hg_ptr  H);

/*
 * Returns zero if the hypergraph was freed successfully.
 */

/****************************************/

/*
 * gst_set_hg_number_of_vertices
 * 
 * Define the number of vertices of a hypergraph.
 */

int gst_set_hg_number_of_vertices (gst_hg_ptr  H,
                                   int         nverts);

/*
 * Returns zero if the number of vertices was set successfully.
 */

/****************************************/

/*
 * gst_set_hg_edges
 * 
 * Define the set of edges of a hypergraph (default associated
 * information). 
 */

int gst_set_hg_edges (gst_hg_ptr  H,
                      int         nedges,
                      int*        edge_sizes,
                      int*        edges,
                      double*     weights);

/*
 * Returns zero if the edges were defined successfully.
 */

/****************************************/

/*
 * gst_set_hg_edge_weights
 * 
 * Set all edge weights of a hypergraph. 
 */

int gst_set_hg_edge_weights (gst_hg_ptr  H,
                             double*     weights);

/*
 * Returns zero if the edges weights were set successfully.
 */

/****************************************/

/*
 * gst_set_hg_vertex_embedding
 * 
 * Embed the vertices in a hypergraph in some $k$-dimensional space.
 * (In the current version only the $2$-dimensional space, the plane, is
 * supported.) 
 */

int gst_set_hg_vertex_embedding (gst_hg_ptr  H,
                                 int         dim,
                                 double*     coords);

/*
 * Returns zero if the vertices were embedded successfully.
 */

/****************************************/

/*
 * gst_set_hg_metric
 * 
 * Set the metric object associated with a hypergraph.
 */

int gst_set_hg_metric (gst_hg_ptr      H,
                       gst_metric_ptr  metric);

/*
 * Returns zero if metric was set successfully.
 */

/****************************************/

/*
 * gst_set_hg_scale_info
 * 
 * Set the scaling information associated with a hypergraph.
 */

int gst_set_hg_scale_info (gst_hg_ptr          H,
                           gst_scale_info_ptr  scinfo);

/*
 * Returns zero if the scaling information was set successfully.
 */

/****************************************/

/*
 * gst_get_hg_terminals
 * 
 * Get terminal vertices for a hypergraph. The terminal indices are
 * returned in the array. 
 */

int gst_get_hg_terminals (gst_hg_ptr  H,
                          int*        nterms,
                          int*        terms);

/****************************************/

/*
 * gst_get_hg_number_of_vertices
 * 
 * Get the number of vertices of a hypergraph.
 */

int gst_get_hg_number_of_vertices (gst_hg_ptr  H);

/*
 * A return value of -1 implies that the hypergraph was invalid.
 */

/****************************************/

/*
 * gst_get_hg_edges
 * 
 * Get the set of edges of a hypergraph. If any of the three final arguments
 * is NULL, the corresponding information is not returned. The
 * user has to allocate space for holding the returned
 * data. Necessary sizes for arrays can be obtained by first
 * obtaining the number of edges, then the edge sizes and finally the
 * vertices for each edge (see example below).
 */

int gst_get_hg_edges (gst_hg_ptr  H,
                      int*        nedges,
                      int*        edge_sizes,
                      int*        edges,
                      double*     weight);

/*
 * Returns zero if the edges were queried successfully.
 * 
 */

/****************************************/

/*
 * gst_get_hg_one_edge
 * 
 * Get information about one edge in the hypergraph. If any of the three
 * last arguments to the function is NULL, the corresponding
 * information is not returned. 
 */

int gst_get_hg_one_edge (gst_hg_ptr  H,
                         int         edge_number,
                         double*     weight,
                         int*        nverts,
                         int*        verts);

/*
 * Returns zero if the edge was queried successfully.
 */

/****************************************/

/*
 * gst_get_hg_vertex_embedding
 * 
 * Get the embedding of the vertices in a hypergraph.
 */

int gst_get_hg_vertex_embedding (gst_hg_ptr  H,
                                 int*        dim,
                                 double*     coords);

/*
 * Returns zero if the embedding was returned successfully.
 */

/****************************************/

/*
 * gst_get_hg_one_vertex_embedding
 * 
 * Return the embedding of a single vertex in a hypergraph.
 */

int gst_get_hg_one_vertex_embedding 
                        (gst_hg_ptr  H,      
                         int         vertex_number,
                         double*     coords);

/*
 * Returns zero if the embedding was returned successfully.
 */

/****************************************/

/*
 * gst_get_hg_edge_embedding
 * 
 * Return the embedding of a subset of edges in a hypergraph. If any
 * of the four last arguments to the function is NULL, the
 * corresponding information is not returned. 
 */

int gst_get_hg_edge_embedding (gst_hg_ptr  H,
                               int         nhgedges,
                               int*        hgedges,
                               int*        nsps,
                               double*     sps,
                               int*        nedges,
                               int*        edges);

/*
 * Returns zero if the embedding was queried successfully.
 * 
 */

/****************************************/

/*
 * gst_get_hg_one_edge_embedding
 * 
 * Return the embedding of a single edge in a hypergraph.
 * Note that the indices of vertices spanned by an edge can be obtained
 * by using gst_get_hg_one_edge().
 */

int gst_get_hg_one_edge_embedding 
                        (gst_hg_ptr  H,
                         int         edge_number,
                         int*        nsps,
                         double*     coords,
                         int*        nedges,
                         int*        edges);

/*
 * Returns zero if embedding was queried successfully.
 * 
 */

/****************************************/

/*
 * gst_get_hg_edge_status
 * 
 * Return the pruning status of an edge.  When _prune_edges
 * runs, it may determine that some edges are ``required'' (such edges
 * must appear in any optimal solution).  It may also determine
 * that certain other edges are ``unneeded'' (at least one optimal
 * solution exists that does not use any ``unneeded'' edge).  By default,
 * edges are neither ``unneeded'' nor ``required.''  It is impossible
 * for an edge to be simultaneously ``unneeded'' and ``required.''
 */

int gst_get_hg_edge_status (gst_hg_ptr  H,
                            int         edge_number,
                            int*        unneeded,
                            int*        required);

/*
 * Returns zero if pruning status was queried successfully.
 * 
 */

/****************************************/

/*
 * gst_get_hg_metric
 * 
 * Get the metric object associated with a hypergraph.
 */

int gst_get_hg_metric (gst_hg_ptr       H,
                       gst_metric_ptr*  metric);

/*
 * Returns zero if the metric was queried successfully.
 */

/****************************************/

/*
 * gst_get_hg_scale_info
 * 
 * Get the scaling information associated with a hypergraph.
 */

int gst_get_hg_scale_info 
                     (gst_hg_ptr           H,
                      gst_scale_info_ptr*  scinfo);

/*
 * Returns zero if the scaling information was queried successfully.
 */

/****************************************/

/*
 * gst_get_hg_properties
 * 
 * Return the list of properties associated with a hypergraph.
 */

gst_proplist_ptr 
        gst_get_hg_properties (gst_hg_ptr  H);

/*
 * Returns the property list.
 */

/****************************************/

/*
 * gst_hg_to_graph
 * 
 * Given a hypergraph having a geometric embedding for each of its vertices
 * and edges, construct an ordinary graph
 * containing the individual edges in the embedding. For a rectilinear
 * embedding the parameter GST_PARAM_GRID_OVERLAY is used to specify
 * that the edges of the reduced grid graph rather than individual edges of the
 * embedding should be returned.
 * 
 * The original vertices in the hypergraph are marked as terminals in
 * the new graph, but the only wayIn a future release of the
 * library, there will be other means of obtaining this information. to
 * get this information out of the new graph is to print it using
 * function gst_save_hg().
 */

gst_hg_ptr gst_hg_to_graph (gst_hg_ptr     H,
                            gst_param_ptr  param, 
                            int*           status);

/*
 * Returns the new graph which represents the embedding.
 */

/****************************************************************/

/*
 * FST generation and pruning functions
 * 
 * _functions
 * 
 * All algorithms for solving geometric Steiner tree problems in
 * use the two-phase approach that consists of full Steiner
 * tree (FST) generation and concatenation.
 * 
 * FST generation is the process of generating a (hopefully small) set
 * of FSTs that is known to contain a Steiner minimum tree (SMT) as a
 * subset. The input to an FST generation algorithm is the set of
 * terminal points, and the output is an embedded hypergraph in which the
 * vertices correspond to terminals and the edges correspond to FSTs. The
 * embedding of each hyperedge (or FST) is the geometric tree structure
 * of the FST.  
 * 
 * In this section we describe the interface to all FST generation
 * algorithms. They are all fairly similar. In addition, a FST 
 * pruning function is given. This function reduces the set of FSTs ---
 * or removes edges from the hypergraph --- such that the resulting
 * hypergraph still contains an SMT. This may speed up the following
 * concatenation algorithm, in particular for very large problem
 * instances. 
 */

/****************************************/

/*
 * gst_generate_fsts
 * 
 * Given a point set (terminals) in the plane, generate a set of FSTs
 * (hyperedges) known to contain an SMT for the point set. The metric
 * that should be used is passed as a parameter (see
 * section~_functions for more on creating metric objects). 
 * The generated FSTs are returned as edges in an embedded hypergraph.  
 */

gst_hg_ptr 
    gst_generate_fsts (int             nterms,
                       double*         terms,
                       gst_metric_ptr  metric,
                       gst_param_ptr   param,
                       int*            status);

/*
 * Returns the resulting FSTs in a hypergraph structure.
 */

/****************************************/

/*
 * gst_generate_efsts
 * 
 * Given a point set (terminals) in the plane, generate a set of FSTs
 * (hyperedges) known to contain an Euclidean SMT for the point
 * set. The FSTs are returned as edges in an embedded hypergraph.
 */

gst_hg_ptr 
    gst_generate_efsts (int            nterms,
                        double*        terms,
                        gst_param_ptr  param,
                        int*           status);

/*
 * Returns the resulting FSTs in a hypergraph structure.
 */

/****************************************/

/*
 * gst_generate_rfsts
 * 
 * Given a point set (terminals) in the plane, generate a set of FSTs
 * (hyperedges) known to contain a rectilinear SMT for the point
 * set. The FSTs are returned as edges in an embedded hypergraph.
 */

gst_hg_ptr 
    gst_generate_rfsts (int            nterms, 
                        double*        terms, 
                        gst_param_ptr  param,
                        int*           status);

/*
 * Returns the resulting FSTs in a hypergraph structure.
 */

/****************************************/

/*
 * gst_generate_ofsts
 * 
 * Given a point set (terminals) in the plane, generate a set of FSTs
 * (hyperedges) known to contain an octilinear SMT for the
 * point set. The FSTs are returned as edges in an embedded
 * hypergraph. 
 */

gst_hg_ptr 
    gst_generate_ofsts (int            nterms, 
                        double*        terms,
                        gst_param_ptr  param, 
                        int*           status);

/*
 * Returns the resulting FSTs in a hypergraph structure.
 */

/****************************************/

/*
 * gst_hg_prune_edges
 * 
 * Given a hypergraph $H$, return a hypergraph $H'$ that has the same
 * vertices as $H$, but a (possibly) reduced set of edges such that there
 * still exists an optimal solution to $H$ in $H'$. The pruning algorithms
 * are metric dependent and require a geometric embedding of the hypergraph
 * vertices and edges.
 */

gst_hg_ptr gst_hg_prune_edges (gst_hg_ptr     H,
                               gst_param_ptr  param, 
                               int*           status);

/*
 * Returns new pruned hypergraph.
 */

/****************************************************************/

/*
 * Hypergraph optimization functions
 * 
 * _functions
 * 
 * The optimization problem associated with hypergraphs is the 
 * minimum spanning tree (MST) in hypergraph problem. Solving this problem
 * solves the FST concatenation problem --- which is the second of the
 * two phases for solving geometric Steiner tree problems. 
 * 
 * The library contains a powerful solver for the general MST in
 * hypergraph problem. This solver uses linear programming and
 * branch-and-cut (or backtrack search for very small problem
 * instances). A large number of parameters can be set to control the
 * solver; consult Appendix~_solver_alg,
 * _stop and _io for a complete list of
 * all solver parameters.
 * 
 * A solution state object has type _solver_ptr. It has an
 * associated hypergraph for which an MST should be found. The solver can
 * be stopped and restarted, e.g., depending on either the quality of
 * (approximate) solutions that are found in the solution process, or on
 * the amount of running time used. The solution state object can contain zero or
 * more feasible (though not necessarily optimal) solutions to the
 * problem. A solution state object refers to both an hypergraph object and a
 * parameter object (from which all necessary parameter values are
 * obtained), as illustrated in Figure~:solutionstate on
 * page~:solutionstate. A demonstration program is given in 
 * Figure~:demo4 on page~:demo4.
 */

/****************************************/

/*
 * gst_create_solver
 * 
 * Create a solution state object for a given hypergraph. The solution
 * process is started by calling the function gst_hg_solve(), and
 * passing the created object as parameter.
 */

gst_solver_ptr 
    gst_create_solver (gst_hg_ptr     H, 
                       gst_param_ptr  param,
                       int*           status);

/*
 * Returns new problem solution state object.
 * 
 * An example is given in Section~_level_interfaces
 * (Figure~:demo4 on page~:demo4).
 */

/****************************************/

/*
 * gst_free_solver
 * 
 * Free a solution state object. All memory associated with this solution
 * state object, except from the associated hypergraph and its objects,
 * are destroyed.
 */

int gst_free_solver (gst_solver_ptr  solver);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 * 
 * An example is given in Section~_level_interfaces
 * (Figure~:demo4 on page~:demo4).
 */

/****************************************/

/*
 * gst_hg_solve
 * 
 * Solve a tree problem for a given hypergraph. In the current version,
 * this function by default computes a minimum spanning tree (MST)
 * in the hypergraph associated with the given solution state object;
 * depending on the parameters given, this function may also compute an
 * heuristic solution to this problem. 
 * 
 * This function can be repeatedly called to solve a (time-consuming)
 * problem, e.g., by setting a CPU time limit for each call. The quality
 * of any solution(s) obtained within the given constraints can be
 * queried by calling gst_get_solver_status().
 */

#define GST_SOLVE_NORMAL		0
#define GST_SOLVE_GAP_TARGET		1
#define GST_SOLVE_LOWER_BOUND_TARGET	2
#define GST_SOLVE_UPPER_BOUND_TARGET	3
#define GST_SOLVE_MAX_BACKTRACKS	4
#define GST_SOLVE_MAX_FEASIBLE_UPDATES	5
#define GST_SOLVE_ABORT_SIGNAL		6
#define GST_SOLVE_TIME_LIMIT		7
#define GST_SOLVE_BB_NODE_LIMIT		8
#define GST_SOLVE_BB_LP_LIMIT		9
#define GST_SOLVE_BB_STOP_REQUESTED	10
#define GST_SOLVE_BACKTRACK_BAD_COSTS	11

int gst_hg_solve (gst_solver_ptr  solver,
                  int *           reason);

/*
 * The function return value indicates whether any serious errors were
 * encountered in the solution process. If this value is zero it means
 * the solver ran successfully and without problems --- although it might
 * have deliberately have been preempted by the user.
 * 
 * A non-zero function return value indicates the error causing the
 * solver to exit prematurely. This could for example be
 * GST_ERR_BACKTRACK_OVERFLOW which can happen if one has set
 * the solver to use backtrack search on an instance which is too big for
 * this purpose (GST_PARAM_SOLVER_ALGORITHM), i.e., more than 32
 * hyperedges. 
 * 
 * When using default parameters (and when not using abort signals) then
 * a value of zero for the parameter means that the solution 
 * search space was completely exhausted. In this case the optimal
 * solution has been found --- unless the problem was found to be infeasible. 
 * However, if the user has set any of the solver stopping condition
 * parameters, such as the CPU time limit, the actual 
 * reason for exiting the solution process is returned using the
 * parameter. Possible return values are one of the
 * following:
 * 
 * 
 * 
 * |ll| 
 * Macro Name                          & Description 
 * GST_SOLVE_NORMAL                 & Normal exit (search space exhausted) 
 * GST_SOLVE_GAP_TARGET            & Requested gap target obtained 
 * GST_SOLVE_LOWER_BOUND_TARGET   & Requested lower bound obtained 
 * GST_SOLVE_UPPER_BOUND_TARGET   & Requested upper bound obtained 
 * GST_SOLVE_MAX_BACKTRACKS        & Max.number of backtracks exceeded 
 * GST_SOLVE_MAX_FEASIBLE_UPDATES & Max.feasible updates exceeded 
 * GST_SOLVE_ABORT_SIGNAL          & Abort signal received 
 * GST_SOLVE_TIME_LIMIT            & CPU time limit exceeded 
 * GST_SOLVE_BB_STOP_REQUESTED	  & Caller requested early termination 
 * GST_SOLVE_BACKTRACK_BAD_COSTS  & Backtrack search does not permit negetive edge costs 
 * 
 * 
 * 
 * An example is given in Section~_level_interfaces
 * (Figure~:demo4 on page~:demo4).
 */

/****************************************/

/*
 * gst_get_solver_status
 * 
 * Return the status of the solution (if any) associated with the given
 * solution state object.  
 */

int gst_get_solver_status (gst_solver_ptr  solver,
                           int*            status);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 * 
 * The value of the parameter is one of the following: 
 * 
 * 
 * 
 * |ll| 
 * Macro Name                 & Description 
 * GST_STATUS_OPTIMAL      & Optimal solution is available  
 * GST_STATUS_INFEASIBLE   & Problem is infeasible 
 * GST_STATUS_FEASIBLE     & Search incomplete, feasible solution(s) known 
 * GST_STATUS_NO_FEASIBLE & Search incomplete, no feasible solutions known 
 * GST_STATUS_NO_SOLUTION & Solver never invoked/hypergraph changed 
 *  
 * 
 * 
 * An example is given in Section~_level_interfaces
 * (Figure~:demo4 on page~:demo4).
 */

/****************************************/

/*
 * gst_get_solver_hypergraph
 * 
 * Return the hypergraph associated with the given solution state object.
 */

gst_hg_ptr gst_get_solver_hypergraph (gst_solver_ptr  solver);

/*
 * Pointer to associated hypergraph object.
 */

/****************************************/

/*
 * gst_get_solver_param
 * 
 * Return the hypergraph associated with the given solution state object.
 */

gst_param_ptr gst_get_solver_param (gst_solver_ptr  solver);

/*
 * Pointer to associated parameter set object.
 */

/****************************************/

/*
 * gst_hg_solution
 * 
 * Retrieve (one of) the best feasible solutions currently known for a
 * given solution state object.
 */

int gst_hg_solution (gst_solver_ptr  solver,
                     int*            nedges,
                     int*            edges,
                     double*         length,
                     int             rank);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 * 
 * The maximal number of feasible solutions that will be retained by
 * the solver is determined by
 * the parameter GST_PARAM_NUM_FEASIBLE_SOLUTIONS. However,
 * for a given solution state object, the actual number of feasible
 * solutions may be less than this maximum --- and even zero.
 * 
 * The function returns GST_ERR_RANK_OUT_OF_RANGE when
 * is less than 0 or greater than or equal to the number of
 * feasible solutions available.
 * 
 */

/****************************************/

/*
 * gst_get_solver_properties
 * 
 * Return the property list associated with a solution state object.
 */

gst_proplist_ptr 
    gst_get_solver_properties (gst_solver_ptr  solver);

/*
 * Returns the property list.
 */

/****************************************************************/

/*
 * Optimization callback functions
 * 
 * 
 * The API provides the ability to invoke user-written code
 * at key points within the internal algorithms.
 * This is accomplished by means of callback functions.
 * The user establishes such a callback function by providing two pointers:
 * 
 *  _func: A pointer to a function to call
 *  _data: A pointer to user-specified data to pass to the
 * 	_func functions
 * 
 * When _func is a NULL pointer (the default), no callback
 * function is invoked.
 * The _data pointer can be anything (in particular, a NULL
 * pointer --- does nothing with this except pass it as an
 * argument to the callback function.
 * 
 * User-written callback functions are passed the following arguments:
 * The argument indicates the location within
 * 's internal algorithms from which the callback
 * function has been invoked.
 * The code in the user's callback function should test this
 * parameter to determine whether that particular calling context
 * requires action.
 * There is a single callback function that is invoked from many
 * places within , but most callback functions are
 * only interested in one (or a small number) of these calling
 * contexts.
 * 
 * The argument is an opaque object that provides
 * access to a variety of internal data that may
 * be of interest to the callback function.
 * The API provides various accessor functions that
 * use this object to access these data.
 * Note that the argument is ephemeral --- the object
 * it refers to exists only during the execution of the
 * callback function.
 * The user must not attempt to use this pointer in any other context
 * (e.g., by storing it in a global variable or in her _data
 * object or other data structure for later use).
 * 
 * The _data argument is the data pointer provided by
 * the user when the callback function was established.
 * 
 * The API defines a macro
 * GST_BB_CALLBACK_ARGS that encapsulates the argument
 * list declarations above.
 * Using this macro helps to automatically update the argument lists
 * of user-written callback functions in the event that future
 * versions of pass additionsl arguments to callback
 * functions.
 * 
 * The argument can have one of the following values:
 * 
 *   GST_CALLBACK_BB_LP_SOLVED:
 * 	Invoked each time a node finishes solving its LP over the
 * 	constraint pool.  (Solving over the constraint pool is an
 * 	iterative process that may invoke the LP solver several
 * 	times until the current LP solution satisfies all
 * 	constraints in the pool.  This callback is only invoked
 * 	when this iteration completes and all pool constraints
 * 	are satisfied.)
 *   GST_CALLBACK_BB_NEW_UPPER_BOUND:
 * 	Invoked whenever an improved upper bound (integer feasible
 * 	solution) is obtained.
 *   GST_CALLBACK_BB_NODE_COMPLETED:
 * 	Invoked when processing of a node has finished.  For
 * 	fractional solutions, this is invoked before selecting
 * 	the branch candidate.
 * 
 */

/* Argument list passed to the bb_callback_func */
#define	GST_BB_CALLBACK_ARGS \
	int wherefrom, gst_node_ptr node, void * cb_data

typedef void gst_bb_callback_func_t (GST_BB_CALLBACK_ARGS);

/* Callback values for wherefrom */

#define GST_CALLBACK_BB_LP_SOLVED		1
#define GST_CALLBACK_BB_NEW_UPPER_BOUND		2
#define GST_CALLBACK_BB_NODE_COMPLETED		3

/****************************************/

/*
 * gst_set_bb_callback_func
 * 
 * Set the branch-and-bound callback function (and data) for the given
 * solution state object to the given function and data pointers.
 */

int gst_set_bb_callback_func (
        gst_solver_ptr    solver,
        void              (*cb_func) (GST_BB_CALLBACK_ARGS),
        void *            cb_data);

/*
 * Returns status code, which is zero upon success.
 */

/****************************************/

/*
 * gst_node_get_solver
 * 
 * A callback node accessor function to get the solution state object
 * on whose behalf this callback function has been invoked.
 */

gst_solver_ptr gst_node_get_solver (gst_node_ptr node);

/*
 * Returns solution state object for which the callback function was
 * invoked.
 */

/****************************************/

/*
 * gst_node_get_z
 * 
 * A callback node accessor function to get the LP objective value
 * associated with the given node.
 */

double gst_node_get_z (gst_node_ptr node);

/*
 * Returns the LP objective value.
 */

/****************************************/

/*
 * gst_node_get_lb_status
 * 
 * A callback node accessor function to get the current node's
 * "lower bound status."
 */

int gst_node_get_lb_status (gst_node_ptr node);

/*
 * Returns an integer having one of the following values:
 * 
 *   GST_LB_STATUS_INFEASIBLE
 *   GST_LB_STATUS_CUTOFF
 *   GST_LB_STATUS_INTEGRAL
 *   GST_LB_STATUS_FRACTIONAL
 *   GST_LB_STATUS_PREEMPTED
 * 
 */

/* Return values from gst_node_get_lb_status */

#define GST_LB_STATUS_INFEASIBLE    1
#define GST_LB_STATUS_CUTOFF        2
#define GST_LB_STATUS_INTEGRAL      3
#define GST_LB_STATUS_FRACTIONAL    4
#define GST_LB_STATUS_PREEMPTED     5

/****************************************/

/*
 * gst_node_get_node_index
 * 
 * A callback node accessor function to get the index of the current node.
 */

int gst_node_get_node_index (gst_node_ptr node);

/*
 * Returns the integer node index.  Index 0 represents the root node,
 * with child nodes having positive indices.  An index of -1 indicates
 * a candidate child node during branch variable selection.
 */

/****************************************/

/*
 * gst_node_get_parent_node_index
 * 
 * A callback node accessor function to get the index of the parent
 * node of the current node.
 */

int gst_node_get_parent_node_index (gst_node_ptr node);

/*
 * Returns the integer index of the parent node.  The parent of the
 * root node has index -1.
 */

/****************************************/

/*
 * gst_node_get_node_depth
 * 
 * A callback node accessor function to get the depth of the current
 * node within the branch-and-bound tree.  The root node has depth
 * zero.
 */

int gst_node_get_node_depth (gst_node_ptr node);

/*
 * Returns the depth of the current node.
 */

/****************************************/

/*
 * gst_node_get_node_branch_var
 * 
 * A callback node accessor function to get the index of the
 * fractional variable that was branched upon to create the current
 * node from its parent.  This is -1 for the root node.
 */

int gst_node_get_node_branch_var (gst_node_ptr node);

/*
 * Returns the branch variable used to create the current node.
 */

/****************************************/

/*
 * gst_node_get_node_branch_direction
 * 
 * A callback node accessor function to get the direction in which
 * the branch variable was constrained to create the current node
 * from its parent.  A value of 0 indicates the var=0 branch.  A
 * value of 1 indicates the var=1 branch.  This is 0 for the root
 * node.
 */

int gst_node_get_node_branch_direction (gst_node_ptr node);

/*
 * Returns the direction of the branch used to create the current node.
 */

/****************************************/

/*
 * gst_node_get_lp_index
 * 
 * A callback node accessor function to get the number of LPs solved
 * at the current node.  (The separation algorithms are run on the
 * solution of each such LP.  If violated constraints are found, they
 * are added to the LP which is then re-solved, which increments this
 * index.)
 */

int gst_node_get_lp_index (gst_node_ptr node);

/*
 * Returns the index of the most recently solved LP at the current
 * node.  The first LP solved receives an index of zero.
 */

/****************************************/

/*
 * gst_node_get_solution
 * 
 * A callback node accessor function to get the LP solution vector
 * associated with the given node.
 */

int gst_node_get_solution (gst_node_ptr node, double * x);

/*
 * Returns an error code, which is zero upon success.
 */

/****************************************/

/*
 * gst_node_get_lb
 * 
 * A callback node accessor function to get the current node's
 * vector of lower bounds.
 */

int gst_node_get_lb (gst_node_ptr node, double * lb);

/*
 * Returns an error code, which is zero upon success.
 */

/****************************************/

/*
 * gst_node_get_ub
 * 
 * A callback node accessor function to get the current node's
 * vector of upper bounds.
 */

int gst_node_get_ub (gst_node_ptr node, double * ub);

/*
 * Returns an error code, which is zero upon success.
 */

/****************************************************************/

/*
 * Message handling functions
 * 
 * _functions
 * All output messages from are passed through
 * user-controllable channels. A given channel may write its output to
 * more than one output (screen/files). Channels have type
 * _channel_ptr. 
 * 
 * In this section we describe the functions for creating and freeing
 * channels, for adding output (screen/files) to a channel, and the basic
 * functions for writing to channels.
 */

/* Flags */
#define GST_CHFLG_POSTSCRIPT                    0x01

typedef struct {
        short           indent;         /* columns, default = 0 */
        short           flags;          /* various options */   
        /* The following items can only be "gotten", not "set" */
        int             column;         /* current column position */
        short           state;          /* various state flags */
        /* Reserved for future use */
        int             reserved1;
        int             reserved2;
} gst_channel_options;

typedef void (*gst_channel_function_ptr) (void *        handle,
                                          const char *  text,
                                          size_t        nbytes);

typedef struct gst_destination *        gst_dest_ptr;

/****************************************/

/*
 * gst_create_channel
 * 
 * Create a channel with an optional set of options. By default, output
 * is unformatted. In the current version, the only formatted output is
 * Postscript; see function gst_channel_setopts() for an example of
 * how to activate Postscript formatting. Consult .h for
 * the detailed structure of _channel_options.
 */

gst_channel_ptr 
    gst_create_channel 
        (const gst_channel_options*  chanopts, 
         int*                        status);

/*
 * Returns the new channel object.
 */

/****************************************/

/*
 * gst_free_channel
 * 
 * Free a channel and all its destinations.
 */

int gst_free_channel (gst_channel_ptr  chan);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_channel_getopts
 * 
 * Get channel options.
 */

int gst_channel_getopts 
        (gst_channel_ptr       chan,
         gst_channel_options*  options);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_channel_setopts
 * 
 * Set channel options.
 */

int gst_channel_setopts 
        (gst_channel_ptr             chan,
         const gst_channel_options*  options);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_channel_add_file
 * 
 * Add a file destination to a channel.
 */

gst_dest_ptr 
    gst_channel_add_file (gst_channel_ptr  chan,
                          FILE*            fp,
                          int*             status);

/*
 * Returns the new destination object (of type _dest_ptr).
 */

/****************************************/

/*
 * gst_channel_add_functor
 * 
 * Add a function as destination to a channel.
 */

typedef size_t 
    gst_channel_func (const char*  buf,
                      size_t       cnt,
                      void*        handle);
gst_dest_ptr 
    gst_channel_add_functor 
                     (gst_channel_ptr    chan,
                      gst_channel_func*  func,
                      void*              handle,
                      int*               status);

/*
 * Returns the new destination object (of type _dest_ptr).
 * 
 */

/****************************************/

/*
 * gst_channel_rmdest
 * 
 * Remove a destination from a channel.
 */

int gst_channel_rmdest (gst_dest_ptr  dest);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_channel_write
 * 
 * Write a string to all destinations in a channel.
 */

int gst_channel_write (gst_channel_ptr  chan,
                       const char*      text,
                       size_t           nbytes);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_channel_printf
 * 
 * Print a formatted string to all destinations in a channel.
 */

int gst_channel_printf (gst_channel_ptr  chan,
                        const char*      format,
                        ...) _GST_PRINTF_ARGS (2,3);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************************************/

/*
 * Input and output functions
 * 
 * _functions
 * A number of functions are provided for input and output of
 * hypergraphs. The input/output format can be chosen using
 * parameters. Scaling information can be associated with input points,
 * and numbers can be printed in unscaled using this information.
 */

/****************************************/

/*
 * gst_create_scale_info
 * 
 * Create a scaling information object.
 */

gst_scale_info_ptr gst_create_scale_info (int* status);

/*
 * Returns the new scaling information object.
 */

/****************************************/

/*
 * gst_free_scale_info
 * 
 * Free a scaling information object.
 */

int gst_free_scale_info (gst_scale_info_ptr scinfo);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_get_points
 * 
 * Reads a point set from a file (e.g., stdin). Point coordinates should be
 * separated by whitespace. Reads until end-of-file or until
 * a specified number of points have been read. 
 * 
 * A scaling information object can be associated with the set of points
 * that are read; if such an object is passed as an argument, this function
 * attempts to find an appropriate scaling for the points to
 * maximize the accuracy of the internal (double) representation. If the
 * scaling information object is NULL, no scaling is performed.
 */

int gst_get_points (FILE*               fp,
                    int                 maxpoints,
                    double**            points,
                    gst_scale_info_ptr  scinfo);

/*
 * Returns the number of read points.
 */

/****************************************/

/*
 * gst_compute_scale_info_digits
 * 
 * Set up various parameters needed for outputting scaled
 * coordinates. Coordinates/distances are printed with the minimum
 * fixed precision whenever this gives the exact result, that is, if all
 * terminal coordinates are integral, they should always be written
 * without a decimal point. Otherwise we will print the
 * coordinates/distances with full precision.
 */

int gst_compute_scale_info_digits 
        (int                 nterms,
         double*             terms,
         gst_scale_info_ptr  scinfo);

/*
 * Returns zero if operation was successful and non-zero
 * otherwise. 
 */

/****************************************/

/*
 * gst_unscale_to_string
 * 
 * Convert a given internal scaled coordinate to a
 * printable unscaled ASCII string.  The internal form is in most
 * cases an integer (to eliminate numeric problems), but the unscaled data 
 * may involve decimal fractions. 
 */

char* gst_unscale_to_string 
          (char*               buffer,
           double              val,
           gst_scale_info_ptr  scinfo);

/*
 * Returns a pointer to a string holding the unscaled value.
 */

/****************************************/

/*
 * gst_unscale_to_double
 * 
 * Convert a given internal form coordinate to an unscaled double.
 */

double gst_unscale_to_double 
           (double              val,
            gst_scale_info_ptr  scinfo);

/*
 * Returns an unscaled double approximation.
 */

/****************************************/

/*
 * gst_load_hg
 * 
 * Load a hypergraph from an input file. The function creates a new
 * hypergraph and adds the vertices and edges read from the input
 * file. The file format must be one of the FST data formats given in
 * Appendix~:fst_formats. 
 */

gst_hg_ptr gst_load_hg (FILE*          fp,
                        gst_param_ptr  param,
                        int*           status);

/*
 * Returns the hypergraph that is read.
 */

/****************************************/

/*
 * gst_save_hg
 * 
 * Print a hypergraph to a file. The print format can be specified by
 * parameter GST_PARAM_SAVE_FORMAT.
 */

int gst_save_hg (FILE*          fp,
                 gst_hg_ptr     H,
                 gst_param_ptr  param);

/*
 * Returns zero if the operation was successful and non-zero
 * otherwise. 
 */

/****************************************************************/

/*
 * Miscellaneous functions
 * 
 * _functions
 * In this section we describe a few miscellaneous functions, e.g.,
 * asynchronous functions that may be used by signal handlers.
 */

/****************************************/

/*
 * gst_deliver_signals
 * 
 * This function is designed to be safely callable from a signal
 * handler. The given signals are delivered to the given solver,
 * which responds to them at some point in the near future.
 * The signals parameter is the bit-wise OR of one or more special
 * signal values defined below.  
 */

void gst_deliver_signals (gst_solver_ptr  solver,
                          int             gstsignals);

/* Signals for gst_deliver_signals */
#define GST_SIG_ABORT           0x0001  /* Abort computation ASAP */
#define GST_SIG_FORCE_BRANCH    0x0002  /* Stop cutting and force a branch */
#define GST_SIG_STOP_TEST_BVAR  0x0004  /* Stop testing branch vars and */
                                        /* use the best one seen so far */
#define GST_SIG_STOP_SEP        0x0008  /* Abort the separation routines */
                                        /* and continue with all cuts */
                                        /* discovered so far */

/*
 * Returns nothing.
 * 
 * The following is a list of possible signals that can be delivered to
 * the solver:
 * 
 * 
 * 
 * |ll| 
 * Macro Name                  & Description 
 * GST_SIG_ABORT            & Abort computation 
 * GST_SIG_FORCE_BRANCH    & Stop cutting and force a branch 
 * GST_SIG_STOP_TEST_BVAR & Stop testing branch variables and 
 *                                   & use the best one seen so far 
 * GST_SIG_STOP_SEP        & Abort the separation routines 
 *                                   & and continue with all cuts 
 *                                   & discovered so far 
 *  
 * 
 */


#ifdef __cplusplus
}
#endif

#endif
