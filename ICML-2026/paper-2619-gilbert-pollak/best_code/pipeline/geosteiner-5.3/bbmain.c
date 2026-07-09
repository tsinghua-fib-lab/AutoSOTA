/***********************************************************************

	$Id: bbmain.c,v 1.71 2022/11/19 13:45:49 warme Exp $

	File:	bbmain.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	The main routine for the branch-and-cut.  (Actually the name
	"bc" is taken so I use "bb" for branch-and-bound.)  It takes
	a file of FSTs and finds the Steiner minimal tree.

************************************************************************

	Modification Log:

	a-1:	11/24/2000	warme
		: Split the main() function off into this new file
		:  so that other programs can call branch-and-cut().
		: Added new switches: -a, -B, -z.
	b-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Removed global variables.
		: New switches: -H -f
		: New flag: -n N (Number of solutions)
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Added include of solver.h.
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.
		: Make features unconditional.
	e-4:	11/18/2022	warme
		: Implement Print_Root_LP functionality using
		:  official callback APIs.
		: Fixed -Wall issues.

************************************************************************/

#include "config.h"
#include "genps.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "solver.h"

#include <ctype.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/*
 * Global Routines
 */

int			main (int, char **);


/*
 * Local Routines
 */

static int		atoi_suf (const char *);
static void		cprintf (char *, ...);
static void		decode_params (int, char **, gst_param_ptr);
static bool		decode_cpu_time_limit (char *, int32u *);
static void		draw_solution (gst_hg_ptr,
				       gst_solver_ptr,
				       bool,
				       gst_channel_ptr,
				       gst_param_ptr);
static void		dump_statistics (gst_hg_ptr,
					 gst_solver_ptr,
					 gst_channel_ptr);
static int		get_int (gst_proplist_ptr plist, int prop_id);
static RETSIGTYPE	handle_signal (int signum);
static void		install_signal_handler (int,
						RETSIGTYPE (*handler)(int));
static void		prescan_params (int, char **);
static void		print_root_lp (GST_BB_CALLBACK_ARGS);
static void		usage (void);

/*
 * Local Variables
 */

static int		Print_Root_LP = FALSE;
static int		Print_FSTs_Only = FALSE;
static int		Print_Title = TRUE;
static gst_solver_ptr	global_solver;

static char *		me;

/*
 * The main routine for the "bb" program.  It takes the output from
 * the "prep" program (phase 1 of our Steiner tree program) and uses
 * a branch-and-cut to find the Steiner minimal tree.
 */

	int
main (

int		argc,
char **		argv
)
{
int			m;
int			p;
int			nverts;
int			status;
int			nsmtfsts;
int			reason;
int			soln_status;
int			slen;
int *			fsts;
bool			is_embedded;
double *		terms;
double			p1time;
char *			descr;
char *			description;
char			tbuf [64];
char			tbuf2 [64];
gst_channel_ptr		chan;
gst_channel_options	chanopts;
gst_hg_ptr		H;
gst_hg_ptr		solhg;
gst_metric_ptr		metric;
gst_param_ptr		params;
gst_proplist_ptr	hgprop;
gst_solver_ptr		solver;
gst_scale_info_ptr	sip;

	me = argv [0];

	setbuf (stdout, NULL);

	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}
	params = gst_create_param (NULL);

	/* Looking for 'FSTs only' flag. This is inefficient but the only way
	   to really stay backwards compatible. */
	prescan_params (argc, argv);

	decode_params (argc, argv, params);

	/* Setup a channel for stdout */
	chan = gst_create_channel (NULL, NULL);
	gst_channel_add_file (chan, stdout, NULL);

	if (NOT Print_FSTs_Only) {
		gst_set_chn_param (params, GST_PARAM_PRINT_SOLVE_TRACE, chan);
	}

	/* Parse the hypergraph from stdin */
	H = gst_load_hg (stdin, NULL, NULL);
	gst_get_hg_scale_info (H, &sip);
	hgprop = gst_get_hg_properties (H);

	is_embedded = FALSE;
	if (NOT Print_FSTs_Only) {
		/* Phase 1 (fst generation) timing information */
		if (gst_get_dbl_property (hgprop,
					  GST_PROP_HG_GENERATION_TIME,
					  &p1time)) {
			gst_channel_printf (chan, " %% Phase 1: Unknown\n");
		}
		else {
			gst_channel_printf (chan, " %% Phase 1: %.2f seconds\n",
					    p1time);
		}

		/* Get the terminals and plot them if they have an embedding */
		gst_get_hg_terminals (H, &nverts, NULL);
		terms = NEWA (2*nverts, double);
		if (NOT gst_get_hg_vertex_embedding (H, NULL, terms)) {
			/* Plot terminals */
			_gst_define_Plot_Terminals (chan, nverts, terms, sip);
			is_embedded = TRUE;
		}
		free (terms);
	}

	/* Trace output should be postscript commented (preceded with %) */
	gst_channel_getopts (chan, &chanopts);
	chanopts.flags |= GST_CHFLG_POSTSCRIPT;
	gst_channel_setopts (chan, &chanopts);

	/* Create solver */
	solver = gst_create_solver (H, params, NULL);

	/* Establish signal handler. */
	global_solver = solver;
	install_signal_handler (SIGTERM, handle_signal);

	if ((NOT Print_FSTs_Only) AND Print_Root_LP) {
		gst_set_bb_callback_func (solver,
					  &print_root_lp,
					  NULL);
	}

	/* Solve the problem */
	status = gst_hg_solve (solver, &reason);

	/* Ignore signals now. */
	install_signal_handler (SIGTERM, SIG_IGN);

	if (status NE 0) {
		if (status EQ GST_ERR_BACKTRACK_OVERFLOW) {
			fprintf (stderr,
				 "%s: Too many vertices or hyperedges\n", me);
		}
		else {
			fprintf (stderr,
				 "%s: Error in solution process (%d)\n",
				 me, status);
		}
		exit (1);
	}

	gst_get_solver_status (solver, &soln_status);
	if (	(soln_status NE GST_STATUS_OPTIMAL)
	    AND (soln_status NE GST_STATUS_FEASIBLE)) {
		fprintf (stderr, "%s: No feasible solutions found.\n", me);
		if (soln_status EQ GST_STATUS_INFEASIBLE) {
			fprintf (stderr, "%s: The problem is infeasible.\n", me);
		}
		exit (1);
	}

	if (gst_get_str_property (hgprop, GST_PROP_HG_NAME, &slen, NULL)) {
		descr = NULL;
	}
	else {
		descr = NEWA (slen + 1, char);
		gst_get_str_property (hgprop, GST_PROP_HG_NAME, NULL, descr);
	}

	/* Get the name of the hypergraph or generate an appropriate name */
	if (   (descr EQ NULL)
	    OR (*descr EQ '\0')) {
		gst_get_hg_metric (H, &metric);
		gst_get_metric_info (metric, &m, &p);

		/* Default */
		description = "%s";

		switch (m) {
		case GST_METRIC_L:
			if (p EQ 1) {
				description = "Rectilinear %s";
			}
			else if (p EQ 2) {
				description = "Euclidean %s";
			}
			break;

		case GST_METRIC_UNIFORM:
			switch (p) {
			case 2:
				description = "Rectilinear %s";
				break;
			case 3:
				description = "Hexagonal %s";
				break;
			case 4:
				description = "Octilinear %s";
				break;
			default:
				sprintf(tbuf, "%d-%%s", p);
				description = tbuf;
				break;
			}
		}

		if (soln_status EQ GST_STATUS_OPTIMAL) {
			sprintf(tbuf2, description, "SMT");
		}
		else {
			sprintf(tbuf2, description, "ST");
		}

		gst_set_str_property (hgprop, GST_PROP_HG_NAME, tbuf2);
	}
	free (descr);

	/* Disable postscript comments */
	gst_channel_getopts (chan, &chanopts);
	chanopts.flags &= ~GST_CHFLG_POSTSCRIPT;
	gst_channel_setopts (chan, &chanopts);

	if (NOT Print_FSTs_Only) {
		draw_solution (H, solver, is_embedded, chan, params);
		dump_statistics (H, solver, chan);
	}

	if (Print_FSTs_Only) {
		/* Get info about the best solution */
		gst_hg_solution (solver, &nsmtfsts, NULL, NULL, 0);
		fsts = NEWA (nsmtfsts, int);
		gst_hg_solution (solver, NULL, fsts, NULL, 0);

		/* Generate a hypergraph only containing the solution edges */
		solhg = gst_create_hg (NULL);
		gst_copy_hg_edges (solhg, H, nsmtfsts, fsts);

		/* Dump it */
		gst_save_hg (stdout, solhg, NULL);

		gst_free_hg (solhg);
		free (fsts);
	}

	/* Clean up */
	gst_free_solver (solver);
	gst_free_hg (H);
	gst_free_channel (chan);
	gst_free_param (params);
	gst_close_geosteiner ();

	CHECK_MEMORY
	exit (0);
}

/*
 * This routine scans the arguments for '-f' flags. If this is present then
 * output should be disabled with the exceptance of a solution hypergraph.
 */
	static
	void
prescan_params (

int		argc,
char **		argv
)
{
char *		ap;
char		c;
char *		p;

	--argc;
	argv++;

	while (argc > 0) {
		ap = *argv++;
		if (*ap EQ '-') {
			++ap;
			while ((c = *ap++) NE '\0') {
				if (c EQ 'f') {
					Print_FSTs_Only = TRUE;
					return;
				}
				if (c EQ 'Z') {
					if (*ap EQ '\0') {
						/* Skip separate parm name */
						++argv;
						--argc;
					}
					/* Skip parm value */
					++argv;
					--argc;
					break;
				}
				for (p = "aBclmnTuz"; *p NE '\0'; p++) {
					if (c EQ *p) break;
				}
				if (c EQ *p) {
					if (*ap EQ '\0') {
						/* Skip separate arg */
						++argv;
						--argc;
					}
					break;
				}
			}
		}
		--argc;
	}
}

/*
 * A conditional printf().
 */
	static
	void
cprintf (

char *format,
...
)
{
va_list		ap;

	if (NOT Print_FSTs_Only) {
		va_start(ap, format);
		vprintf(format, ap);
		va_end(ap);
	}
}

/*
 * This routine decodes the various command-line arguments.
 */
	static
	void
decode_params (

int		argc,
char **		argv,
gst_param_ptr	params
)
{
int		i;
int		j;
char *		ap;
char		c;
char *		pname;
char *		buffer;
char *		s;
int		nmerge;
int		max_merge;
int32u		cpu_time_limit;
int		branch_var_policy;
int		bvp_min;
int		bvp_max;
int		rv;
char *		merge_files [32];

	--argc;
	me = *argv++;

	nmerge = 0;
	max_merge = sizeof (merge_files) / sizeof (merge_files [0]);

	cprintf (" %% %s\n", me);
	cprintf (" %% Args:\n");

	while (argc > 0) {
		ap = *argv++;
		if (*ap NE '-') {
			usage ();
		}
		cprintf (" %%	%s\n", ap);
		++ap;
		while ((c = *ap++) NE '\0') {
			switch (c) {
			case '2':
				gst_set_int_param (params, GST_PARAM_SEED_POOL_WITH_2SECS,
							   GST_PVAL_SEED_POOL_WITH_2SECS_DISABLE);
				break;

#ifdef CPLEX
			case 'a':
				if ((*ap NE '\0') OR (argc <= 1)) {
					usage ();
				}
				ap = *argv++;
				cprintf (" %%	%s\n", ap);
				--argc;
				gst_set_int_param (params, GST_PARAM_CPLEX_MIN_ROWS, atoi_suf (ap));
				ap = *argv++;
				cprintf (" %%	%s\n", ap);
				--argc;
				gst_set_int_param (params, GST_PARAM_CPLEX_MIN_NZS, atoi_suf (ap));
				ap = "";
				break;
#endif

			case 'b':
				gst_set_int_param (params, GST_PARAM_BRANCH_VAR_POLICY,
						   GST_PVAL_BRANCH_VAR_POLICY_WEAK);
				break;

			case 'B':
				{
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				branch_var_policy = atoi_suf (ap);
				gst_query_int_param (params,
						     GST_PARAM_BRANCH_VAR_POLICY,
						     NULL, NULL, &bvp_min, &bvp_max);
				if ((branch_var_policy < bvp_min) OR
				    (branch_var_policy > bvp_max)) {
					fprintf (stderr,
						"%s: Invalid branch variable"
						" policy `%s'."
						"  Valid policies range"
						" from %d to %d.\n",
						me,
						ap,
						bvp_min,
						bvp_max);
					usage ();
				}
				gst_set_int_param (params,
						   GST_PARAM_BRANCH_VAR_POLICY,
						   branch_var_policy);
				ap = "";
				}
				break;

			case 'c':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				gst_set_str_param (params, GST_PARAM_CHECKPOINT_FILENAME, ap);
				ap = "";
				break;

			case 'f':
				break;

			case 'l':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}

				if (NOT decode_cpu_time_limit (ap, &cpu_time_limit)) {
					usage ();
				}

				gst_set_dbl_param (params, GST_PARAM_CPU_TIME_LIMIT, (double)cpu_time_limit);

				ap = "";
				break;

			case 'H':
				gst_set_int_param (params,
						   GST_PARAM_SOLVER_ALGORITHM,
						   GST_PVAL_SOLVER_ALGORITHM_BACKTRACK_SEARCH);
				break;


			case 'L':
				gst_set_int_param (params,
						   GST_PARAM_LOCAL_CUTS_MODE,
						   GST_PVAL_LOCAL_CUTS_MODE_SUBTOUR_RELAXATION);
				gst_set_int_param (params,
						   GST_PARAM_LOCAL_CUTS_MAX_DEPTH,
						   GST_PVAL_LOCAL_CUTS_MAX_DEPTH_ANYLEVEL);
				break;

			case 'm':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				if (nmerge + 1 >= max_merge) {
					fprintf (stderr, "%s: too many -m args!\n", me);
					exit (1);
				}
				if (strchr (ap, ':') NE NULL) {
					fprintf (stderr,
						 "%s: Error: -m argument \"%s\" has a ':'"
						 " character in it, which is not permitted.\n",
						 me, ap);
					exit (1);
				}
				merge_files [nmerge++] = ap;
				ap = "";
				break;

			case 'n':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				gst_set_int_param (params, GST_PARAM_NUM_FEASIBLE_SOLUTIONS, atoi (ap));
				ap = "";
				break;

#ifdef LPSOLVE
			case 'p':
				gst_set_int_param (params,
						   GST_PARAM_LP_SOLVE_PERTURB,
						   GST_PVAL_LP_SOLVE_PERTURB_ENABLE);
				break;
#endif

			case 'r':
				Print_Root_LP = TRUE;
				break;

			case 'R':
				gst_set_int_param (params,
						   GST_PARAM_CHECK_ROOT_CONSTRAINTS,
						   GST_PVAL_CHECK_ROOT_CONSTRAINTS_ENABLE);
				break;

#ifdef LPSOLVE
			case 's':
				gst_set_int_param (params,
						   GST_PARAM_LP_SOLVE_SCALE,
						   GST_PVAL_LP_SOLVE_SCALE_ENABLE);
				break;
#endif
			case 't':
				Print_Title = FALSE;
				break;

			case 'T':
				{
				int check_branch_vars_thoroughly;
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				check_branch_vars_thoroughly = atoi_suf (ap);
				if (check_branch_vars_thoroughly < 1) {
					usage ();
				}
				gst_set_int_param (params,
						   GST_PARAM_CHECK_BRANCH_VARS_THOROUGHLY,
						   check_branch_vars_thoroughly);
				ap = "";
				}
				break;

			case 'u':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				gst_set_dbl_param (params, GST_PARAM_INITIAL_UPPER_BOUND, atof (ap));
				ap = "";
				break;

			case 'z':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				gst_set_int_param (params, GST_PARAM_TARGET_POOL_NON_ZEROS, atoi_suf (ap));
				ap = "";
				break;

			case 'Z':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					cprintf (" %%	%s\n", ap);
					--argc;
				}
				pname = ap;
				if (argc <= 1) {
					usage ();
				}
				ap = *argv++;
				cprintf (" %%	%s\n", ap);
				--argc;

				rv = gst_set_param (params, pname, ap);
				switch (rv) {
				case 0:
					/* The parameter was correctly set */
					break;
				case GST_ERR_UNKNOWN_PARAMETER_ID:
					fprintf(stderr,
					    "Parameter '%s' does not exist.\n",
					    pname);
					usage ();
					break;
				case GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE:
					fprintf(stderr,
					    "Parameter value, %s, for '%s' is out of range.\n",
					    ap, pname);
					usage ();
					break;
				default:
					usage ();
				}

				ap = "";
				break;

			default:
				usage ();
				break;
			}
		}
		--argc;
	}

	if (nmerge > 0) {
		/* Convert merge pathnames into a single string containing a	*/
		/* colon-separated list of pathnames.  This is how the library	*/
		/* expects this parameter to be encoded.			*/
		j = 1;				/* space for final null. */
		for (i = 0; i < nmerge; i++) {
			if (i > 0) {
				++j;		/* add space for colon. */
			}
			j += strlen (merge_files [i]);
		}
		buffer = NEWA (j, char);
		ap = buffer;
		for (i = 0; i < nmerge; i++) {
			if (i > 0) {
				*ap++ = ':';
			}
			s = merge_files [i];
			for (;;) {
				c = *s++;
				if (c EQ '\0') break;
				*ap++ = c;
			}
		}
		*ap++ = '\0';

		gst_set_str_param (params, GST_PARAM_MERGE_CONSTRAINT_FILES, buffer);

		free (buffer);
	}
}


/*
 * This routine prints out the proper usage and exits.
 */

static char *	arg_doc [] = {
	"",
	"\t-2\tOmit all 2-terminal SECs from the initial",
	"\t\t constraint pool.",
#ifdef CPLEX
	"\t-a M N\tForce CPLEX allocation to be at least M",
	"\t\t rows and N non-zeros.",
#endif
	"\t-b\tDisable \"strong branching\", which chooses",
	"\t\tbranching variables very carefully.",
	"\t-B N\tSet branch variable selection policy.",
	"\t\t N=0: naive max of mins,",
	"\t\t N=1: smarter lexicographic max of mins (default),",
	"\t\t N=2: product of improvements.",
	"\t-c P\tPathname of checkpoint file to restore (if",
	"\t\t present) and/or update.  The files are actually",
	"\t\t named P.chk and P.ub, with temporary files named",
	"\t\t P.tmp, P.new and P.nub.",
	"\t-f\tDump no other information than the FSTs in the",
	"\t\t solution.",
	"\t-H\tAlways use backtrack search instead of branch and cut.",
	"\t\t This is faster for small problem instances.",
	"\t-l T\tTerminate run after T CPU time is expended.",
	"\t\t T can be in days, hours, minutes and/or seconds",
	"\t\t (as shown below).",
	"\t-L\tEnable local cuts.",
	"\t-m P\tMerge constraints from checkpoint file P with",
	"\t\t those of the formulation.",
#ifdef LPSOLVE
	"\t-p\tUse perturbations when solving LP's.",
#endif
	"\t-r\tPrint root LP relaxation, if fractional.",
	"\t-R\tWhen optimal root LP relaxation is obtained,",
	"\t\tdetermine for each LP iteration the number of",
	"\t\tfinal constraints whose first violation occurred",
	"\t\tduring that iteration.",
#ifdef LPSOLVE
	"\t-s\tUse scaling when solving LP's.",
#endif
	"\t-n N\tOutput the N best solutions (default: 1)",
	"\t-t\tDo not display the title (name, length, time)",
	"\t-T N\tSearch N times more thoroughly for strong",
	"\t\t branching variables.",
	"\t-u B\tSets the initial upper bound to B.",
	"\t-z N\tSets the target number of pool non-zeros to N.",
	"\t-Z P V\tSet parameter P to value V.",
	"",
	"\tExample CPU times are:",
	"\t\t-l 3days2hours30minutes15seconds",
	"\t\t-l1000seconds -l1000 -l 2h30m",
	"",
	NULL
};

	static
	void
usage (void)

{
char **		pp;
char *		p;

	(void) fprintf (stderr,
			"\nUsage: %s"
			" [-2bL"
#ifdef LPSOLVE
			"p"
#endif
			"rR"
#ifdef LPSOLVE
			"s"
#endif
			"t"
			"]"
#ifdef CPLEX
			" [-a minNumRows minNumNonZeros]"
#endif
			" [-B branch_var_policy]"
			" [-c checkpoint_file]"
			" [-l cpu-time-limit]"
			" [-m merge_checkpoint_file]"
			" [-n N]"
			" [-T N]"
			" [-u upper-bound]"
			" [-z N]"
			" [-Z parameter value]"
			" <phase-1-data-file\n",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}

/*
 * Convert a decimal string to an integer, and permit various suffixes,
 * such as 'k' = 1000, 'K' = 1024, etc.
 */

	static
	int
atoi_suf (

const char *	s		/* IN - string to convert */
)
{
int		c;
int		sign;
int		num;

	do {
		c = *s++;
	} while (isspace (c));

	sign = 1;
	if (c EQ '-') {
		sign = -1;
	}

	num = 0;
	while ((c >= '0') AND (c <= '9')) {
		num = 10 * num + (c - '0');
		c = *s++;
	}
	switch (c) {
	case '\0':				break;
	case 'k':	num *= 1000;		break;
	case 'K':	num *= 1024;		break;
	case 'm':	num *= 1000000;		break;
	case 'M':	num *= (1024 * 1024);	break;
	default:
		fprintf (stderr, "%s: Unknown numeric suffix '%c'.\n",
			 me, c);
		exit (1);
	}

	return (sign * num);
}

/*
 * This routine decodes an argument string that represents an amount of
 * CPU time.  The time is the sum of a sequence of time groups.  Each
 * time group is a sequence of decimal digits followed by an optional
 * letter defining units of minutes, hours, days, etc.
 * The regular expression accepted is:
 *
 * ([0-9]+([dhms][a-z]*)?)*
 */

	static
	bool
decode_cpu_time_limit (

char *		ap,
int32u *	limit
)
{
int32u		total;
int32u		group;
char		c;

	*limit = 0;
	total = 0;
	c = *ap++;
	while (c NE '\0') {
		if ((c < '0') OR (c > '9')) {
			return (FALSE);
		}
		group = c - '0';
		while (((c = *ap++) NE '\0') AND ('0' <= c) AND (c <= '9')) {
			group = (10 * group) + (c - '0');
		}
		switch (c) {
		case 'd':	group *= (24 * 60 * 60);	break;
		case 'h':	group *= (60 * 60);		break;
		case 'm':	group *= 60;			break;
		case 's':					break;
		case '\0':					break;
		default:	return (FALSE);
		}
		/* strip rest of unit name... */
		while ((c > 0) AND ((c < '0') OR (c > '9'))) {
			c = *ap++;
		}
		total += group;
	}

	*limit = total;

	return (TRUE);

}

/*
 * This routine plots the LP relaxation solution for the root node.  We
 * do this whenever -r is specified and we get an optimal LP relaxation
 * solution that is fractional.
 */
	static
	void
print_root_lp (

GST_BB_CALLBACK_ARGS	/* wherefrom, node, cb_data */
)
{
int			nverts, nedges, node_idx, lb_status;
double			root_time;
double			root_len;
char			buf1 [32];
char			title [256];
gst_solver_ptr		solver;
gst_hg_ptr		H;
gst_param_ptr		params;
double *		frac_sol;
gst_scale_info_ptr	sip;
gst_proplist_ptr	solprop;
gst_channel_ptr		chan;
gst_channel_options	oldopts;
gst_channel_options	newopts;

	if (wherefrom NE GST_CALLBACK_BB_NODE_COMPLETED) {
		return;
	}

	node_idx = gst_node_get_node_index (node);
	if (node_idx NE 0) {
		/* Not the root node. */
		return;
	}

	/* The root node has finished.  We only care about	*/
	/* fractional solutions.				*/
	lb_status = gst_node_get_lb_status (node);
	if (lb_status NE GST_LB_STATUS_FRACTIONAL) {
		return;
	}

	/* We have a fractional solution.  Generate plot of the	*/
	/* fractional solution.					*/

	solver	= gst_node_get_solver (node);
	H	= gst_get_solver_hypergraph (solver);
	params	= gst_get_solver_param (solver);

	gst_get_hg_scale_info (H, &sip);
	gst_get_hg_terminals (H, &nverts, NULL);
	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);

	frac_sol = NEWA (nedges, double);
	gst_node_get_solution (node, frac_sol);

	solprop = gst_get_solver_properties (solver);
	gst_get_dbl_property (solprop, GST_PROP_SOLVER_ROOT_TIME, &root_time);
	gst_get_dbl_property (solprop, GST_PROP_SOLVER_ROOT_LENGTH, &root_len);

	gst_unscale_to_string (buf1, root_len, sip);

	sprintf (title,
		 "Root Node:  %d points,  LP %d,  length = %s,  %.2f seconds",
		 nverts,
		 0,
		 /* nodep -> iter, */
		 buf1,
		 root_time);

	gst_get_chn_param (params,
			   GST_PARAM_PRINT_SOLVE_TRACE,
			   &chan);

	gst_channel_getopts (chan, &oldopts);
	newopts = oldopts;
	newopts.flags &= ~GST_CHFLG_POSTSCRIPT;
	newopts.indent = 0;
	gst_channel_setopts (chan, &newopts);

	_gst_plot_lp_solution (chan, H, title, frac_sol, BIG_PLOT);

	gst_channel_setopts (chan, &oldopts);

	free (frac_sol);
}

/*
 *  Drawing solution(s) to the given channel.
 */
	void
draw_solution (

gst_hg_ptr		H,		/* IN - the hypergraph... */
gst_solver_ptr		solver,		/* IN - and the solver */
bool			is_embedded,	/* IN - the solution can be drawn */
gst_channel_ptr		chan,
gst_param_ptr		params
)
{
int			i;
int			j;
int			k;
int			fst_num;
int			fst_nterms;
int			nsmtfsts;
int			nsps;
int			nsol;
int			nverts;
int			slen;
double			p1time;
double			p2time;
int *			fsts;
int *			term_index;
double			length;
double			weight;
double *		sps;
char			title[256];
char			buf1[32];
char			buf2[32];
char *			description;
gst_scale_info_ptr	sip;
gst_proplist_ptr	hgprop;
gst_proplist_ptr	solprop;

	gst_get_hg_scale_info (H, &sip);

	if (is_embedded) {
		/* Print out a certificate of the solution.  This	*/
		/* consists of the coordinates of each of the Steiner	*/
		/* points.						*/
		gst_hg_solution (solver, &nsmtfsts, NULL, NULL, 0);
		fsts = NEWA(nsmtfsts, int);
		gst_hg_solution (solver, NULL, fsts, NULL, 0);

		gst_get_hg_edge_embedding (H, nsmtfsts, fsts, &nsps,
					   NULL, NULL, NULL);
		if (nsps NE 0) {
			sps = NEWA (2*nsps, double);
			gst_get_hg_edge_embedding (H, nsmtfsts, fsts,
						   NULL, sps, NULL, NULL);

			gst_channel_printf (chan, "\n %% Certificate of solution:\n");
			for (i = 0; i < nsps; i++) {
				gst_unscale_to_string (buf1, sps[2*i], sip);
				gst_unscale_to_string (buf2, sps[2*i+1], sip);
				gst_channel_printf (chan, " %% @C\t%s\t%s\n",
						    buf1, buf2);
			}
			free (sps);
		}
		free (fsts);
	}

	/* Get various info from the hypergraph and the solver */
	hgprop	= gst_get_hg_properties (H);
	solprop	= gst_get_solver_properties (solver);
	gst_get_str_property (hgprop, GST_PROP_HG_NAME, &slen, NULL);
	description = NEWA (slen + 1, char);
	gst_get_str_property (hgprop, GST_PROP_HG_NAME, NULL, description);
	gst_get_hg_terminals (H, &nverts, NULL);
	p1time = 0.0; p2time = 0.0;
	gst_get_dbl_property (hgprop,  GST_PROP_HG_GENERATION_TIME, &p1time);
	gst_get_dbl_property (solprop, GST_PROP_SOLVER_CPU_TIME, &p2time);

	/* Print out the fsts in the K best solutions */
	gst_get_int_param (params, GST_PARAM_NUM_FEASIBLE_SOLUTIONS, &nsol);
	for (k = 0; k < nsol; k++) {
		gst_hg_solution (solver, &nsmtfsts, NULL, &length, k);
		fsts = NEWA(nsmtfsts, int);
		gst_hg_solution (solver, NULL, fsts, NULL, k);

		if (is_embedded) {
			gst_unscale_to_string (buf1, length, sip);
			if (k EQ 0) {
				sprintf (title,
					 "%s:  %d points,  length = %s,  "
					 "%.2f seconds",
					 description, nverts, buf1,
					 p1time + p2time);
			}
			else {
				sprintf (title, "Solution %d,  length = %s",
					 k+1, buf1);
			}

			/* Plot solution */
			_gst_overlay_plot_subset (chan, H, Print_Title ? title:NULL,
						  nsmtfsts, fsts, BIG_PLOT);
		}
		else {
			/* Just print the hyperedges */
			for (i = 0; i < nsmtfsts; i++) {
				fst_num = fsts[i];
				gst_channel_printf (chan, "Edge %d: ", fst_num);
				gst_get_hg_one_edge (H, fst_num, &weight, &fst_nterms, NULL);
				term_index = NEWA (fst_nterms, int);
				gst_get_hg_one_edge (H, fst_num, NULL, NULL, term_index);

				for (j = 0; j < fst_nterms; j++) {
					gst_channel_printf (chan, "%d ", term_index[j]);
				}
				gst_unscale_to_string (buf1, weight, sip);
				gst_channel_printf (chan, "%s\n", buf1);

				free (term_index);
			}
			gst_channel_printf (chan, "\n");
		}
		free (fsts);
	}
	free (description);
}

/*
 * This routine dumps a wide range of statistics about the solution. Most
 * of the information is obtained through solver 'properties'.
 */

#define GETINT(a) (get_int (solprop, a))

	void
dump_statistics (

gst_hg_ptr		H,	/* IN - the hypergraph... */
gst_solver_ptr		solver,	/* IN - and the solver */
gst_channel_ptr		chan
)
{
int			i;
int			j;
int			nt;
int			nfsts;
int			edge10;
int			total_edge_count;
int			max_edge_size;
int			nsmtfsts;
int			nverts;
int			slen;
int *			edge_count;
int *			edge_sizes;
int *			fsts;
double			length;
double			rlength;
double			mstlength;
double			redmst;
double			p1time;
double			p2time;
double			rtime;
char			buf1[32];
char			buf2[32];
char			buf3[32];
char *			description;
gst_proplist_ptr	hgprop;
gst_proplist_ptr	solprop;
gst_scale_info_ptr	sip;

	gst_get_hg_scale_info (H, &sip);

	gst_get_hg_terminals (H, &nverts, NULL);

	gst_hg_solution (solver, &nsmtfsts, NULL, &length, 0);
	fsts = NEWA (nsmtfsts, int);
	gst_hg_solution (solver, NULL, fsts, NULL, 0);

	/* Get all hyperedges/fsts */
	gst_get_hg_edges (H, &nfsts, NULL, NULL, NULL);
	edge_sizes = NEWA (nfsts, int);
	gst_get_hg_edges (H, NULL, edge_sizes, NULL, NULL);

	/* Tally various statistics about the solution.		*/
	edge_count = NEWA (nverts + 1, int);
	total_edge_count = 0;
	max_edge_size = 0;
	for (i = 0; i <= nverts; i++) {
		edge_count [i] = 0;
	}

	for (i = 0; i < nsmtfsts; i++) {
		nt = edge_sizes [fsts[i]];

		++(edge_count [nt]);
		total_edge_count += nt;
		if (nt > max_edge_size) {
			max_edge_size = nt;
		}
	}

	/* Problem summary... */
	hgprop	= gst_get_hg_properties (H);
	solprop	= gst_get_solver_properties (solver);

	gst_get_str_property (hgprop, GST_PROP_HG_NAME, &slen, NULL);
	description = NEWA (slen + 1, char);
	gst_get_str_property (hgprop, GST_PROP_HG_NAME, NULL, description);
	gst_channel_printf (chan, "%% @0 %s\n",
		description NE NULL ? description : "");

	p1time = 0.0; p2time = 0.0;
	gst_get_dbl_property (hgprop,  GST_PROP_HG_GENERATION_TIME, &p1time);
	gst_get_dbl_property (solprop, GST_PROP_SOLVER_CPU_TIME, &p2time);

	gst_channel_printf (chan, "%% N M Nodes LPs P1CPU P2CPU TotCPU\n");
	gst_channel_printf (chan, "%% @1 %d %d %d %d %.2f %.2f %.2f\n",
		nverts,
		nfsts,
		GETINT (GST_PROP_SOLVER_NUM_NODES),
		GETINT (GST_PROP_SOLVER_NUM_LPS),
		p1time, p2time, p1time + p2time);

	/* Solution and root node statistics... */
	gst_unscale_to_string (buf1, length, sip);
	if (gst_get_dbl_property (solprop, GST_PROP_SOLVER_ROOT_LENGTH, &rlength)) {
		sprintf (buf2, "-");
		sprintf (buf3, "-");
	}
	else {
		if (GETINT (GST_PROP_SOLVER_ROOT_OPTIMAL)) {
			sprintf (buf2, "%.6f", rlength);
		}
		else {
			sprintf (buf2, "(%.6f)", rlength);
		}

		sprintf (buf3, "%7.5f", 100.0 * (length - rlength) / length);
	}

	if (gst_get_dbl_property (hgprop, GST_PROP_HG_MST_LENGTH, &mstlength)) {
		redmst = 0.0;
	}
	else {
		redmst = 100.0 * (mstlength - length) / mstlength;
	}

	rtime = 0.0;
	gst_get_dbl_property (solprop, GST_PROP_SOLVER_ROOT_TIME, &rtime);

	gst_channel_printf (chan, "%% Z RootZ %%Gap RootLPs RootCPU RedMST\n");
	gst_channel_printf (chan, "%% @2 %s %s %s %d %.2f %.4f\n",
		buf1,
		buf2,
		buf3,
		GETINT (GST_PROP_SOLVER_ROOT_LPS),
		rtime,
		redmst);

	/* Initial constraint pool statistics... */
	gst_channel_printf (chan, "%% InitPRows InitPNZ InitLPRows InitLPNZ\n");
	gst_channel_printf (chan, "%% @3 %d %d %d %d\n",
		GETINT (GST_PROP_SOLVER_INIT_PROWS),
		GETINT (GST_PROP_SOLVER_INIT_PNZ),
		GETINT (GST_PROP_SOLVER_INIT_LPROWS),
		GETINT (GST_PROP_SOLVER_INIT_LPNZ));

	/* Root constraint pool statistics... */
	gst_channel_printf (chan, "%% RootPRows RootPNZ RootLPRows RootLPNZ\n");
	gst_channel_printf (chan, "%% @4 %d %d %d %d\n",
		GETINT (GST_PROP_SOLVER_ROOT_PROWS),
		GETINT (GST_PROP_SOLVER_ROOT_PNZ),
		GETINT (GST_PROP_SOLVER_ROOT_LPROWS),
		GETINT (GST_PROP_SOLVER_ROOT_LPNZ));

	/* Final constraint statistics... */
	gst_channel_printf (chan, "%% FinalPRows FinalPNZ FinalLPRows FinalLPNZ\n");
	gst_channel_printf (chan, "%% @5 %d %d %d %d\n",
		GETINT (GST_PROP_SOLVER_FINAL_PROWS),
		GETINT (GST_PROP_SOLVER_FINAL_PNZ),
		GETINT (GST_PROP_SOLVER_FINAL_LPROWS),
		GETINT (GST_PROP_SOLVER_FINAL_LPNZ));

	/* Statistics on the SMT: number of FSTs, size and distribution. */

	edge10 = 0;
	gst_channel_printf (chan, "%% SMTFSTs SMTAvgFSTSz SMTMaxFSTSz #2FSTs #3FSTs ... #10FSTS #>10FSTs\n");
	gst_channel_printf (chan, "%% @6 %d %f %d",
		nsmtfsts,
		((double) total_edge_count) / ((double) nsmtfsts),
		max_edge_size);
	for (i = 2; i <= 10; i++) {
		j = (i <= nverts) ? edge_count [i] : 0;
		edge10 += j;
		gst_channel_printf (chan, " %d", j);
	}
	gst_channel_printf (chan, " %d\n", nsmtfsts - edge10);

	free (description);
	free (edge_sizes);
	free (edge_count);
	free (fsts);
}

/*
 * Get the value of an integer property.
 */

	static
	int
get_int (

gst_proplist_ptr	plist,		/* IN - property list */
int			prop_id		/* IN - property to retrieve */
)
{
int		value;

	if (gst_get_int_property (plist, prop_id, &value) NE 0) {
		return (0);
	}
	return (value);
}

/*
 * Install a signal handler for the given signal.  We use sigaction()
 * if we have it, or signal() if we do not.
 */

	void
install_signal_handler (

int		sig_num,		/* IN - signal number */
RETSIGTYPE	(*handler) (int)	/* IN - handler function */
)
{
#ifdef HAVE_SIGACTION
struct sigaction	sigact;

	memset (&sigact, 0, sizeof (sigact));
	sigact.sa_handler = handler;
	sigaction (sig_num, &sigact, NULL);
#else
	signal (sig_num, handler);
#endif
}

/*
 * A handler for SIGTERM.  We poke the solver to do something else.
 */

	static
	RETSIGTYPE
handle_signal (

int	signum
)
{
	gst_deliver_signals (global_solver, GST_SIG_FORCE_BRANCH);

#ifndef HAVE_SIGACTION
	/* Using signal().  Assume we need to re-establish the handler. */
	install_signal_handler (signum, handle_signal);
#endif
}
