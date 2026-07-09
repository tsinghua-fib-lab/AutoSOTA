/***********************************************************************

	$Id: egmp.c,v 1.14 2022/11/19 13:45:51 warme Exp $

	File:	egmp.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 2000, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Support routines for the EFST generator that use the
	GNU Multi-Precision arithmetic library (GMP -- if we have
	it) to compute certain items with high accuracy.

************************************************************************

	Modification Log:

	b-1:	11/25/2000	warme
		: Created.
	b-2:	02/02/2014	warme
		: Fix hard coded dependence upon limb bits.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Implement new cost_extension object for
		:  SteinLib "integer" format.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix -Wall issues.

************************************************************************/

#include "egmp.h"

#include "config.h"

#if defined(HAVE_GMP)

#include <stdio.h>
#include <gmp.h>

#include "costextension.h"
#include "efst.h"
#include "fatal.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "metric.h"
#include "parmblk.h"
#include <stdlib.h>
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

double		_gst_compute_EFST_length (struct einfo *	eip,
					  struct eqp_t *	eqpt);
void		_gst_qr3_clear (qr3_t * p);
void		_gst_qr3_init (qr3_t * p);
void		_gst_update_eqpoint_and_displacement (
					struct einfo *		eip,
					struct eqp_t *		eqpk);

struct cost_extension *
		_gst_egmp_graph_cost_extension (
					struct gst_hypergraph *	cip,
					struct gst_param *	params);

/*
 * Local Equates
 */

#define DEBUG_PRINT	0

struct ExEqp;
struct GceInfo;

/*
 * Local Routines
 */

static void	_egmp_cext_free (struct cost_extension *	extp);
static void	_egmp_cext_write_edge_cost (
				struct cost_extension *		extp,
				FILE *				fp,
				int				i);
static void	allocate_graph_arrays (struct GceInfo * gcp);
static void	compute_edges_for_fst (struct GceInfo *		gcp,
				       mpq_srcptr		scale_factor,
				       int			fstidx);
static void	compute_eqpoint (struct qr3_point *	p,
				 struct einfo *		eip,
				 struct eqp_t *		eqpk);
static void	compute_qr3_length_squared (qr3_t *		result,
					    struct qr3_point *	p1,
					    struct qr3_point *	p2);
static void	compute_scale_factor (mpq_ptr		scale,
				      struct GceInfo *	gcp);
static void	compute_sqrt_a (mpf_ptr res, mpq_srcptr a);
static void	compute_sqrt_a_plus_b_root3 (mpf_ptr		res,
					     mpq_srcptr		a,
					     mpq_srcptr		b);
static void	compute_sqrt_b_root3 (mpf_ptr res, mpq_srcptr b);
static void	compute_sqrt_scaled_qr3 (mpf_ptr	res,
					 const qr3_t *	A,
					 mpq_srcptr	scale);
static void	count_graph_items (struct GceInfo * gcp);
static void	free_graph_arrays (struct GceInfo * gcp);
static void	print_qr3 (const qr3_t * x);
static void	process_fst (struct GceInfo * gcp, int i);
static void	project_qr3_point (struct qr3_point *		S,
				   const struct qr3_point *	E,
				   const struct qr3_point *	C,
				   const struct qr3_point *	T);
static void	qr3_div (qr3_t * RES, const qr3_t * A, const qr3_t * B);
static void	qr3_dot_product (qr3_t *			RES,
				 const struct qr3_point *	A,
				 const struct qr3_point *	B);
static void	qr3_mul (qr3_t * dst, const qr3_t * p1, const qr3_t * p2);
static double	qr3_to_double (struct einfo *, qr3_t *);
static void	r_to_q (mpq_t q_dst, double r_src);
static void	traverse_compute_exact_steiner_points (
				struct GceInfo *	gcp,
				struct full_set *	fsp,
				struct qr3_point *	tpnt,
				struct ExEqp *		r);
static struct ExEqp *	traverse_fst_graph (
				struct GceInfo *	gcp,
				struct full_set *	fsp,
				int			v1,
				int			e1,
				int			v2);

/*
 * A routine to recompute the coordinates of the given eq-point and its
 * corresponding displacement vector.  Using purely floating point
 * arithmetic, errors tend to accumulate as the eq-point order grows.
 * Once we are pretty sure that we are going to save this eq-point, we
 * call this routine that re-computes these quantities correct to within
 * 1/2 ULP of the floating point arithmetic -- thus preventing the
 * accumulation of such errors.
 */

	void
_gst_update_eqpoint_and_displacement (

struct einfo *		eip,		/* IN - EFST generation info */
struct eqp_t *		eqpk		/* IN - eq-point to recompute */
)
{
double			nx;
double			ny;
mpq_t			rtmp;
struct point *		tp;
struct qr3_point *	ep;
qr3_t			dv_tmp;

	mpq_init (rtmp);
	_gst_qr3_init (&dv_tmp);

	ep = &(eip -> cur_eqp);

	compute_eqpoint (ep, eip, eqpk);

#if DEBUG_PRINT
	printf ("\nPoint %3d: original = (%24.20f, %24.20f)\n",
		eqpk -> index,
		eqpk -> E.x,
		eqpk -> E.y);
#endif

	nx = qr3_to_double (eip, &(ep -> x));
	ny = qr3_to_double (eip, &(ep -> y));

#if DEBUG_PRINT
	printf ("\tNew:          (%24.20f, %24.20f)\n", nx, ny);

	{	double	ex, ey;
		ex = fabs (nx - eqpk -> E.x) / nx;
		ey = fabs (ny - eqpk -> E.y) / ny;

		printf ("\tErrors:\t%d\t(%14g, %14g)\n", eqpk -> S, ex, ey);
	}

	printf ("\tSymbolic: X = ");
	mpz_out_str (stdout, 10, mpq_numref (ep -> x.a));
	printf (" / ");
	mpz_out_str (stdout, 10, mpq_denref (ep -> x.a));
	printf (" + ");
	mpz_out_str (stdout, 10, mpq_numref (ep -> x.b));
	printf (" * sqrt (3) / ");
	mpz_out_str (stdout, 10, mpq_denref (ep -> x.b));
	printf ("\n");

	printf ("\tSymbolic: Y = ");
	mpz_out_str (stdout, 10, mpq_numref (ep -> y.a));
	printf (" / ");
	mpz_out_str (stdout, 10, mpq_denref (ep -> y.a));
	printf (" + ");
	mpz_out_str (stdout, 10, mpq_numref (ep -> y.b));
	printf (" * sqrt (3) / ");
	mpz_out_str (stdout, 10, mpq_denref (ep -> y.b));
	printf ("\n");
#endif

	eqpk -> E.x = nx;
	eqpk -> E.y = ny;

	tp = &(eip -> eqp [eqpk -> origin_term].E);

	r_to_q (rtmp, tp -> x);
	mpq_sub (dv_tmp.a, ep -> x.a, rtmp);
	mpq_set (dv_tmp.b, ep -> x.b);
	eqpk -> DV.x = qr3_to_double (eip, &dv_tmp);

	r_to_q (rtmp, tp -> y);
	mpq_sub (dv_tmp.a, ep -> y.a, rtmp);
	mpq_set (dv_tmp.b, ep -> y.b);
	eqpk -> DV.y = qr3_to_double (eip, &dv_tmp);

	_gst_qr3_clear (&dv_tmp);
	mpq_clear (rtmp);
}

/*
 * Compute the exact coordinates of the given eq-point as a member
 * of the field Q(sqrt(3)) -- i.e., as (A + B*sqrt(3), C + D*sqrt(3))
 * where A, B, C and D are exact rational numbers.
 */

	void
compute_eqpoint (

struct qr3_point *	out,		/* OUT - eq-point coordinates */
struct einfo *		eip,		/* IN - EFST generation info */
struct eqp_t *		eqpk		/* IN - eq-point to calculate */
)
{
struct qr3_point	P, Q, R;
mpq_t			c2, c3, t1, t2;

	if (eqpk -> L EQ NULL) {
		/* Base case -- a terminal. */

		r_to_q (out -> x.a, eqpk -> E.x);
		r_to_q (out -> y.a, eqpk -> E.y);

		mpq_set_ui (out -> x.b, 0, 1);
		mpq_set_ui (out -> y.b, 0, 1);

		return;
	}

	/* Recurse, let P be the right point, and Q the left eq-point. */

	_gst_qr3_init (&P.x);
	_gst_qr3_init (&P.y);
	_gst_qr3_init (&Q.x);
	_gst_qr3_init (&Q.y);
	_gst_qr3_init (&R.x);
	_gst_qr3_init (&R.y);

	compute_eqpoint (&P, eip, eqpk -> R);
	compute_eqpoint (&Q, eip, eqpk -> L);

	mpq_sub (R.x.a, Q.x.a, P.x.a);
	mpq_sub (R.x.b, Q.x.b, P.x.b);
	mpq_sub (R.y.a, Q.y.a, P.y.a);
	mpq_sub (R.y.b, Q.y.b, P.y.b);

	mpz_init_set_ui (mpq_numref (c2), 2);
	mpz_init_set_ui (mpq_denref (c2), 1);

	mpz_init_set_ui (mpq_numref (c3), 3);
	mpz_init_set_ui (mpq_denref (c3), 1);

	mpq_init (t1);
	mpq_init (t2);

	mpq_mul (t1, c3, R.y.b);
	mpq_sub (t2, R.x.a, t1);
	mpq_div (t1, t2, c2);
	mpq_add (out -> x.a, P.x.a, t1);

	mpq_sub (t1, R.x.b, R.y.a);
	mpq_div (t2, t1, c2);
	mpq_add (out -> x.b, P.x.b, t2);

	mpq_mul (t1, c3, R.x.b);
	mpq_add (t2, R.y.a, t1);
	mpq_div (t1, t2, c2);
	mpq_add (out -> y.a, P.y.a, t1);

	mpq_add (t1, R.y.b, R.x.a);
	mpq_div (t2, t1, c2);
	mpq_add (out -> y.b, P.y.b, t2);

	mpq_clear (t2);
	mpq_clear (t1);
	mpq_clear (c3);
	mpq_clear (c2);

	_gst_qr3_clear (&R.y);
	_gst_qr3_clear (&R.x);
	_gst_qr3_clear (&Q.y);
	_gst_qr3_clear (&Q.x);
	_gst_qr3_clear (&P.y);
	_gst_qr3_clear (&P.x);
}

/*
 * Convert a double to a rational number.
 */

	static
	void
r_to_q (

mpq_t		q_dst,		/* OUT - rational number */
double		r_src		/* IN  - double to convert */
)
{
int		mant_size;
int		expon;
mpf_t		x;
mpz_t		tmp;
mpz_ptr		np;
mpz_ptr		dp;

	/* First, convert to MP-floating point. */
	mpf_init2 (x, 64);
	mpf_set_d (x, r_src);

	/* Access mantissa as an mpz_t... */
	tmp [0]._mp_alloc	= x [0]._mp_prec + 1;
	tmp [0]._mp_size	= x [0]._mp_size;
	tmp [0]._mp_d		= x [0]._mp_d;

	np = mpq_numref (q_dst);
	dp = mpq_denref (q_dst);

	/* Since mantissa is fractional (not an integer), we	*/
	/* must include the mantissa size in the exponent...	*/
	mant_size = x [0]._mp_size;
	if (mant_size < 0) {
		mant_size = -mant_size;
	}
	expon = x [0]._mp_exp - mant_size;
	if (expon < 0) {
		/* Set numerator */
		mpz_set (np, tmp);

		/* Divide by limb-base**(- expon) */
		mpz_set_ui (dp, 1);
		mpz_mul_2exp (dp, dp, GMP_LIMB_BITS * (- expon));
	}
	else {
		/* Set denominator */
		mpz_set_ui (dp, 1);

		/* Set numerator to proper shift factor. */
		mpz_set_ui (np, 1);
		mpz_mul_2exp (np, np, GMP_LIMB_BITS * expon);
		/* multiply mantissa in... */
		mpz_mul (np, np, tmp);
	}

	mpq_canonicalize (q_dst);

	mpf_clear (x);
	/* DO NOT clear tmp! */
}

/*
 * Initialize an Q(sqrt(3)) number.
 */

	void
_gst_qr3_init (

qr3_t *		p
)
{
	mpq_init (p -> a);
	mpq_init (p -> b);
}


/*
 * Clear a Q(sqrt(3)) number.
 */

	void
_gst_qr3_clear (

qr3_t *		p
)
{
	mpq_clear (p -> a);
	mpq_clear (p -> b);
}

/*
 * Accurately translate an element Z of Q(sqrt(3)) into a double.
 * Let Z = A + B * sqrt(3), where A and B are rational.
 *
 *	Z - A = B * sqrt(3)	==>	Z^2 - 2*A*Z + A^2 - 3*B^2 = 0
 *
 * We solve start with a floating point approximation of Z, convert
 * it to a rational and then apply Newton iterations until we
 * achieve the desired precision.  The termination test is as
 * follows:
 *
 *	|Z - (A + B*sqrt(3))| < eps * (A + B*sqrt(3)),
 *
 *	Z^2 - 2*(A + B*sqrt(3))*Z + A^2 + 2*A*B*sqrt(3) + 3*B^2
 *		< eps^2 * (A^2 + 2*A*B*sqrt(3) + 3*B^2),
 *
 *	Z^2 - 2*A*Z + A^2 + 3*B^2 - eps^2*(A^2 + 3*B^2)
 *			< (2*B*Z - (1 - eps^2) 2*A*B) * sqrt(3),
 *
 *	Z^2 - 2*A*Z + (1 - eps^2)*(A^2 + 3*B^2)
 *			< (Z - (1 - eps^2)*A) * 2 * B * sqrt(3),
 *
 * The obvious thing here is to square both sides and compare.
 * However, one must be careful regarding the signs of the two
 * sides when doing this.
 */

	static
	double
qr3_to_double (

struct einfo *	eip,		/* IN - global EFST info */
qr3_t *		p		/* IN - A + B*sqrt(3) to convert */
)
{
int		flag;
mpq_t		z;
mpq_t		t1;
mpq_t		t2;
mpq_t		t3;
mpq_t		t4;
mpq_t		t5;
mpq_t		t6;
mpq_t		c_1_minus_eps2;
double		xf;

#define EPS	64		/* relative error of 2^(-EPS) */

	if (mpz_sgn (mpq_numref (p -> b)) EQ 0) {
		return (mpq_get_d (p -> a));
	}

	mpq_init (z);
	mpq_init (t1);
	mpq_init (t2);
	mpq_init (t3);
	mpq_init (t4);
	mpq_init (t5);
	mpq_init (t6);
	mpq_init (c_1_minus_eps2);

	/* c_1_minus_eps2 is now 0 / 1.  Set it to be	*/
	/* 1 - 2**( - 2 * EPS).				*/

	mpz_mul_2exp (mpq_denref (c_1_minus_eps2),
		      mpq_denref (c_1_minus_eps2),
		      2*EPS);
	mpz_sub_ui (mpq_numref (c_1_minus_eps2),
		    mpq_denref (c_1_minus_eps2),
		    1);

	/* Compute t1 = (A^2 - 3*B^2) */

	mpq_mul (t3, p -> a, p -> a);
	mpq_mul (t4, p -> b, p -> b);
	mpz_mul_ui (mpq_numref (t4), mpq_numref (t4), 3);
	mpq_canonicalize (t4);
	mpq_sub (t1, t3, t4);

	/* Compute t2 = (1 - eps^2) * (A^2 + 3*B^2).	*/

	mpq_add (t2, t3, t4);
	mpq_mul (t2, t2, c_1_minus_eps2);

	/* Compute t3 = (1 - eps^2)*A.	*/

	mpq_mul (t3, c_1_minus_eps2, p -> a);

	/* Constants for the termination test are now ready.	*/
	/* Time to start up the Newton iteration.		*/

	/* Approximate solution using floating point as a start... */

	xf = mpq_get_d (p -> a) + mpq_get_d (p -> b) * sqrt(3.0);

	/* Put into rational form. */
	r_to_q (z, xf);

	for (;;) {
		/* Compute Z = new Z, via Newton */
		mpq_mul (t5, z, z);
		mpq_sub (t5, t5, t1);
		mpq_sub (t6, z, p -> a);
		mpz_mul_ui (mpq_numref (t6), mpq_numref (t6), 2);
		mpq_canonicalize (t6);
		mpq_div (z, t5, t6);

#if DEBUG_PRINT
		printf ("	New z is %24.20f\n", mpq_get_d (z));
#endif

		if (eip -> params -> multiple_precision <= 1) {
			/* Termination test takes time.  At level 1,	*/
			/* we just do 1 Newton iteration and quit.	*/
			break;
		}

		/* Now test termination... */
		mpq_set (t6, p -> a);
		mpz_mul_ui (mpq_numref (t6), mpq_numref (t6), 2);
		mpq_canonicalize (t6);
		mpq_sub (t5, z, t6);
		mpq_mul (t5, t5, z);
		mpq_add (t5, t5, t2);

		mpq_sub (t6, z, t3);
		mpq_mul (t6, t6, p -> b);

		/* Time to check the signs... */

		flag = 1;
		if (mpz_sgn (mpq_numref (t5)) < 0) {
			if (mpz_sgn (mpq_numref (t6)) >= 0) break;
			flag = -1;
		}
		else if (mpz_sgn (mpq_numref (t6)) < 0) {
			continue;
		}

		mpq_mul (t5, t5, t5);
		mpq_mul (t6, t6, t6);
		mpz_mul_ui (mpq_numref (t6), mpq_numref (t6), 12);
		mpq_canonicalize (t6);

#if DEBUG_PRINT
		printf ("	t5 = %24g, t6 = %24g\n",
			mpq_get_d (t5),
			mpq_get_d (t6));
#endif

		if (mpq_cmp (t5, t6) * flag < 0) break;
	}

	xf = mpq_get_d (z);

	mpq_clear (c_1_minus_eps2);
	mpq_clear (t6);
	mpq_clear (t5);
	mpq_clear (t4);
	mpq_clear (t3);
	mpq_clear (t2);
	mpq_clear (t1);
	mpq_clear (z);

	return (xf);
}

/*
 * Routine to compute the length of a given EFST (i.e., Simpson line)
 * to within 1/2 ULP.  (Modulo good behavior of the sqrt() function...)
 *
 * The exact position of the eq-point end of the Simpson line has already
 * been stored in eip -> cur_eqp.
 */

	double
_gst_compute_EFST_length (

struct einfo *		eip,		/* IN - EFST generation info */
struct eqp_t *		eqpt		/* IN - terminal end of Simpson line */
)
{
qr3_t			x;
qr3_t			y;
double			len;

	_gst_qr3_init (&x);
	_gst_qr3_init (&y);

	r_to_q (x.a, - eqpt -> E.x);
	r_to_q (y.a, - eqpt -> E.y);

	mpq_add (x.a, x.a, eip -> cur_eqp.x.a);
	mpq_add (y.a, y.a, eip -> cur_eqp.y.a);
	mpq_set (x.b, eip -> cur_eqp.x.b);
	mpq_set (y.b, eip -> cur_eqp.y.b);

	qr3_mul (&x, &x, &x);
	qr3_mul (&y, &y, &y);

	mpq_add (x.a, x.a, y.a);
	mpq_add (x.b, x.b, y.b);

	len = sqrt (qr3_to_double (eip, &x));

	_gst_qr3_clear (&y);
	_gst_qr3_clear (&x);

	return (len);
}

/*
 * Multiply two elements of Q(sqrt(3)).
 */

	static
	void
qr3_mul (

qr3_t *			dst,		/* OUT - destination */
const qr3_t *		p1,		/* IN - first operand */
const qr3_t *		p2		/* IN - second operand */
)
{
mpq_t		res_a;
mpq_t		tmp1;
mpq_t		tmp2;

	mpq_init (res_a);
	mpq_init (tmp1);
	mpq_init (tmp2);

	mpq_mul (tmp1, p1 -> a, p2 -> a);
	mpq_mul (tmp2, p1 -> b, p2 -> b);
	mpz_mul_ui (mpq_numref (tmp2), mpq_numref (tmp2), 3);
	mpq_canonicalize (tmp2);
	mpq_add (res_a, tmp1, tmp2);

	mpq_mul (tmp1, p1 -> a, p2 -> b);
	mpq_mul (tmp2, p1 -> b, p2 -> a);
	mpq_add (dst -> b, tmp1, tmp2);

	mpq_set (dst -> a, res_a);

	mpq_clear (tmp2);
	mpq_clear (tmp1);
	mpq_clear (res_a);
}

/*
 * Various temporary data needed by _gst_egmp_graph_cost_extension(),
 * and the various routines it calls.
 */

struct GceInfo {
	struct gst_hypergraph *
			cip;		/* The EFSTs we are processing */
	struct gst_param *
			params;		/* Algorithm parameters */
	int		num_bits;	/* Desired precision of final	*/
					/* integer edge costs */
	int		fp_numbits;	/* Number of bits to use for */
					/* high-precision floating-point */
					/* numbers */
	int		num_fsts;	/* Number of FSTs */
	int		num_terms;	/* Number of terminals */
	int		num_steins;	/* Number of Steiner points */
	int		num_edges;	/* Number of "intra FST" edges */
	int		max_terms;	/* Max num terminals in any FST */
	int		max_steins;	/* Max num Steiner points in any FST */
	int		max_edges;	/* Max num edges in any FST */
	int		scale_pow;	/* Power of 10 we are scaling by */
	int *		stein_base;	/* Base Steiner point index for */
					/* each FST */
	int *		edge_base;	/* Base edge index for each FST */
	int *		count;		/* To count degree of each vertex */
	int *		start;		/* First adjacency for each vertex */
	int *		ptrs;		/* For filling in adjacencies */
	int *		adj_edge;	/* Index of incident edge */
	int *		adj_vert;	/* Index of incident vertex */
	qr3_t *		fst_qr3;	/* Length *squared* of each FST, */
					/* in exact ration Q(sqrt(3)) form */
	qr3_t *		edge_qr3;	/* Length *squared* of each edge, */
					/* in exact ration Q(sqrt(3)) form */
	mpf_ptr		fst_mpf;	/* Length of each FST, in high- */
					/* precision floating-point form */
	mpf_ptr		edge_mpf;	/* Length of each intra-FST edge */
					/* in high-precision floating-point */
					/* form */
	mpz_ptr		weight;		/* Final integer edge weights */
	struct ExEqp *	eqpts;		/* Array of "exact" eq-points */
	double		max_relerr;	/* Maximum relative error (between */
					/* exact versus floating-point) */
					/* among all edges */
	int		max_relerr_edge;
					/* Edge that achieved this maximum */
					/* relative error */
	int		max_relerr_fst;	/* FST that achieved this maximum */
					/* relative error */
};

/*
 * A version of "struct eqp_t" that is specialized for exact computation.
 */

struct ExEqp {
	int			index;	/* Index of this point */
	bool			term;	/* If true, point is a terminal */
	struct ExEqp *		L;	/* Pointer to left eq-point */
	struct ExEqp *		R;	/* Pointer to right eq-point */
	struct point		Efp;	/* Approximate (floating-point) */
					/* coordinates of terminal or */
					/* Steiner point.  (This is only */
					/* used to determine proper ordering */
					/* of L R subtrees.) */
	struct qr3_point	E;	/* Coordinates of eq-point */
	struct qr3_point	DC;	/* Center of eq-point circle */
	struct qr3_point	STP;	/* Coordinates of Steiner point */
};


/*
 * Our special "derived class" implementation of "struct cost_extension".
 * The object here is to encapsulate an array of GMP integers, one per
 * edge -- and to print the arbitrary integer edge cost when required.
 */

struct _egmp_cost_extension {
	struct cost_extension	base;		/* Base class */

	int		num_edges;		/* Number of edges */
	mpz_ptr		edge_costs;		/* Cost of each edge */
};

/*
 * Virtual table used for this derived class.
 */

static const struct cost_ext_vtab	_egmp_cext_vtab = {
	_egmp_cext_free,
	_egmp_cext_write_edge_cost,
};

/*
 * Return a special "cost extension" object for the "graph" version of
 * the given Euclidean FSTs.  This extension object contains an array
 * of edge costs stored as GMP integers, allowing the values to contain
 * as many significant bits as desired.
 *
 * We scale the actual edge weights by a suitable factor 10**M, so that
 * the scaled edge weights (properly rounded to nearest integer values)
 * fit nicely into K-bit unsigned integers.
 *
 * Note: K is actually one of the algorithmic *parameters* that the user
 * can vary.  (Must be >= 32, default is 64.)
 *
 * It is not enough for each individual integer weight to fit into
 * K bits -- the whole point is that the resulting instance should be
 * correctly solvable using K-bit integer arithmetic.  In particular, we
 * would like all feasible integer solutions to have objective values that
 * fit within an unsigned K-bit integer.
 *
 * Theorem: Given a Euclidean hypergraph computed by our own EFST generator,
 * the longest feasible integer solution is the EMST, consisting of all
 * edges of cardinality 2.
 *
 * Proof: Our EFST generator produces hypergraphs whose only edges of
 * cardinality 2 are the MST edges, and all generated 2-edges produce a
 * unique EMST.  Every edge of cardinality J >= 3 does not violate the
 * bottleneck Steiner distances among its terminals.  Consequently, any
 * feasible combination of cardinality J >= 3 edges yields a partial
 * solution no worse than the BMST, which is no worse than the MST.
 * Finishing the solution with MST edges maintains this state.  QED
 *
 * Note: all bets are off if the hypergraph came from some other source.
 * Furthermore, there may be infeasible subsets of edges (e.g., cyclic,
 * or other non-tree structures) whose total edge weight exceeds the
 * precision of an unsigned K-bit integer.  You have been warned!
 */

	struct cost_extension *
_gst_egmp_graph_cost_extension (

struct gst_hypergraph *	cip,	/* IN: EFSTs to compute extended costs for */
struct gst_param *	params	/* IN: algorithm parameters */
)
{
int				i, n;
int				num_bits, fp_numbits;
struct _egmp_cost_extension *	extp;
mpz_ptr				weight;
struct GceInfo			gcinfo;
mpq_t				scale_factor, scale2;
bitmap_t *			fset_mask;

	FATAL_ERROR_IF ((cip EQ NULL) OR
			(cip -> full_trees EQ NULL) OR
			(NOT _gst_is_euclidean (cip)));

	/* Get the number of bits of precision desired. */
	num_bits = params -> save_int_numbits;

	memset (&gcinfo, 0, sizeof (gcinfo));

	gcinfo.cip		= cip;
	gcinfo.params		= params;
	gcinfo.num_bits		= num_bits;
	gcinfo.max_relerr	= -1.0;
	gcinfo.max_relerr_edge	= -1;
	gcinfo.max_relerr_fst	= -1;

	/* Compute number of bits to use in high-precision float-point. */
	fp_numbits = 2 * num_bits;
	if (fp_numbits < 256) {
		fp_numbits = 256;
	}
	gcinfo.fp_numbits = fp_numbits;

	count_graph_items (&gcinfo);

	allocate_graph_arrays (&gcinfo);

	fset_mask = cip -> initial_edge_mask;
	n = gcinfo.num_fsts;
	for (i = 0; i < n; i++) {
		process_fst (&gcinfo, i);
	}

	/* Compute the proper rational scale factor (10**M/1 or 1/10**M	). */
	/* This is the desired power of 10 that maked the scaled edges	   */
	/* fit nicely within the desired K-bit integer size.		   */

	mpq_init (scale_factor);
	mpq_init (scale2);

	compute_scale_factor (scale_factor, &gcinfo);

	/* Compute scale_factor**2.  This is how we should scale	*/
	/* numbers that are length**2 values before conversion into	*/
	/* length**1 values.						*/
	mpq_mul (scale2, scale_factor, scale_factor);

	for (i = 0; i < n; i++) {
		if (BITON (fset_mask, i)) {
			compute_edges_for_fst (&gcinfo, scale2, i);
		}
	}

	fprintf (stderr,
		 "Scaling factor: 10**%d\n"
		 "Maximum relative error: %.17g\n"
		 " on edge %d (FST %d, edge %d)\n",
		 gcinfo.scale_pow,
		 gcinfo.max_relerr,
		 gcinfo.max_relerr_edge,
		 gcinfo.max_relerr_fst,
		 gcinfo.max_relerr_edge
		 - gcinfo.edge_base [gcinfo.max_relerr_fst]);

	/* Get our array of integer edge weights. */
	weight = gcinfo.weight;
	gcinfo.weight = NULL;

	/* Clean up other storage. */
	mpq_clear (scale2);
	mpq_clear (scale_factor);
	free_graph_arrays (&gcinfo);

	extp = NEW (struct _egmp_cost_extension);
	extp -> base.vtab = &_egmp_cext_vtab;
	extp -> num_edges = gcinfo.num_edges;
	extp -> edge_costs = weight;

	return (&(extp -> base));
}

/*
 * Count the various graph items, terminals, FSTs, Steiner points, edges.
 */

	static
	void
count_graph_items (

struct GceInfo *	gcp		/* IN/OUT: global data */
)
{
int			i, nt, ns, ne;
int			num_terms, num_fsts, num_steins, num_edges;
int			max_terms, max_steins, max_edges;
struct gst_hypergraph *	cip;
struct full_set *	fsp;
bitmap_t *		fset_mask;

	cip = gcp -> cip;

	/* Get size of the graph we are producing. */
	num_terms	= cip -> num_verts;
	num_fsts	= cip -> num_edges;
	fset_mask	= cip -> initial_edge_mask;

	num_steins	= 0;
	num_edges	= 0;
	max_terms	= 0;
	max_steins	= 0;
	max_edges	= 0;

	for (i = 0; i < num_fsts; i++) {
		fsp = cip -> full_trees [i];
		nt = fsp -> terminals -> n;
		ns = 0;
		if (fsp -> steiners NE NULL) {
			ns = fsp -> steiners -> n;
		}
		ne = fsp -> nedges;

		/* Verify fundmental relationships. */
		FATAL_ERROR_IF ((nt < 2) OR
				(nt > num_terms) OR
				(ns NE nt - 2) OR
				(ne NE nt + ns - 1));

		/* Only count these for active FSTs. */
		if (BITON (fset_mask, i)) {
			num_steins += ns;
			num_edges  += ne;
		}

		if (nt > max_terms) {
			max_terms = nt;
		}
		if (ns > max_steins) {
			max_steins = ns;
		}
		if (ne > max_edges) {
			max_edges = ne;
		}
	}

	gcp -> num_fsts		= num_fsts;
	gcp -> num_terms	= num_terms;
	gcp -> num_steins	= num_steins;
	gcp -> num_edges	= num_edges;
	gcp -> max_terms	= max_terms;
	gcp -> max_steins	= max_steins;
	gcp -> max_edges	= max_edges;
}

/*
 * Allocate and initialize various arrays for the graph structures.
 */

	static
	void
allocate_graph_arrays (

struct GceInfo *	gcp		/* IN/OUT: global data */
)
{
int			i, j, k, n;
int			nv, ne;
int			fp_numbits, num_fsts, num_edges;
struct gst_hypergraph *	cip;
int *			stein_base;
int *			edge_base;
struct full_set *	fsp;
qr3_t *			fst_qr3;
qr3_t *			edge_qr3;
mpf_ptr			fst_mpf;
mpf_ptr			edge_mpf;
mpz_ptr			weight;
struct ExEqp *		eqpts;
struct ExEqp *		eqpk;
int *			ip;
bitmap_t *		fset_mask;

	cip		= gcp -> cip;
	fp_numbits	= gcp -> fp_numbits;
	num_fsts	= gcp -> num_fsts;
	num_edges	= gcp -> num_edges;
	fset_mask	= cip -> initial_edge_mask;

	/* Allocate properly sized arrays, and initialize. */
	stein_base = NEWA (num_fsts, int);
	edge_base  = NEWA (num_fsts, int);
	j = 0;
	k = 0;
	for (i = 0; i < num_fsts; i++) {
		if (NOT (BITON (fset_mask, i))) {
			stein_base [i]	= -1;
			edge_base [i]	= -1;
			continue;
		}
		fsp = cip -> full_trees [i];
		stein_base [i] = j;
		edge_base [i]  = k;
		j += fsp -> steiners -> n;
		k += fsp -> nedges;
	}

	fst_qr3 = NEWA (num_fsts, qr3_t);
	for (i = 0; i < num_fsts; i++) {
		_gst_qr3_init (&fst_qr3 [i]);
	}

	edge_qr3 = NEWA (num_edges, qr3_t);
	for (i = 0; i < num_edges; i++) {
		_gst_qr3_init (&edge_qr3 [i]);
	}

	fst_mpf = NEWA (num_fsts, __mpf_struct);
	for (i = 0; i < num_fsts; i++) {
		mpf_init2 (&fst_mpf [i], fp_numbits);
	}

	edge_mpf = NEWA (num_edges, __mpf_struct);
	for (i = 0; i < num_edges; i++) {
		mpf_init2 (&edge_mpf [i], fp_numbits);
	}

	nv = gcp -> max_terms + gcp -> max_steins;
	ne = gcp -> max_edges;

	eqpts = NEWA (nv, struct ExEqp);
	memset (eqpts, 0, nv * sizeof (eqpts [0]));
	for (k = 0; k < nv; k++) {
		eqpk = &eqpts [k];
		_gst_qr3_init (&(eqpk -> E.x));
		_gst_qr3_init (&(eqpk -> E.y));
		_gst_qr3_init (&(eqpk -> DC.x));
		_gst_qr3_init (&(eqpk -> DC.y));
		_gst_qr3_init (&(eqpk -> STP.x));
		_gst_qr3_init (&(eqpk -> STP.y));
	}

	weight = NEWA (num_edges, __mpz_struct);
	for (i = 0; i < num_edges; i++) {
		mpz_init (&weight [i]);
	}

	gcp -> stein_base	= stein_base;
	gcp -> edge_base	= edge_base;
	gcp -> fst_qr3		= fst_qr3;
	gcp -> edge_qr3		= edge_qr3;
	gcp -> fst_mpf		= fst_mpf;
	gcp -> edge_mpf		= edge_mpf;
	gcp -> weight		= weight;
	gcp -> eqpts		= eqpts;

	n = nv + (nv + 1) + nv + 4 * ne;
	ip = NEWA (n, int);
	gcp -> count	= ip;		ip += nv;
	gcp -> start	= ip;		ip += (nv + 1);
	gcp -> ptrs	= ip;		ip += nv;
	gcp -> adj_edge	= ip;		ip += (2 * ne);
	gcp -> adj_vert	= ip;		ip += (2 * ne);
}

/*
 * Process the i-th FST.  More specifically, we compute the following:
 *
 *	- The exact length *squared* of the FST in Q(sqrt(3)) form.
 *	- The exact length *squared* of each edge witin the FST,
 *	  also in Q(sqrt(3)) form.
 *
 * In order to do this, we must first reconstruct the FST in the form
 * of a tree of equilateral-points.
 */

	static
	void
process_fst (

struct GceInfo *	gcp,		/* IN/OUT: global data */
int			i		/* IN: index of FST to process */
)
{
int			j, k, n;
int			nt, ns, ne;
int			v1, e1, v2;
struct gst_hypergraph *	cip;
struct full_set *	fsp;
struct pset *		terms;
struct pset *		steins;
struct edge *		edges;
struct edge *		ep;
struct ExEqp *		eqpk;
struct qr3_point *	qp1;
struct qr3_point *	qp2;
int *			count;
int *			start;
int *			ptrs;
int *			adj_edge;
int *			adj_vert;

	cip	= gcp -> cip;
	fsp	= cip -> full_trees [i];

	terms	= fsp -> terminals;
	steins	= fsp -> steiners;
	edges	= fsp -> edges;

	count		= gcp -> count;
	start		= gcp -> start;
	ptrs		= gcp -> ptrs;
	adj_edge	= gcp -> adj_edge;
	adj_vert	= gcp -> adj_vert;

	nt = terms -> n;
	ns = 0;
	if (steins NE NULL) {
		ns = steins -> n;
	}
	ne = fsp -> nedges;

	/* Initialize the eq-point for each terminal. */
	for (k = 0; k < nt; k++) {
		eqpk = &(gcp -> eqpts [k]);

		eqpk -> index	= k;
		eqpk -> term	= TRUE;

		/* Set the "approximate" (floating-point) eq-point coords. */
		eqpk -> Efp = terms -> a [k];

		/* Set exact eq-point / terminal coordinates. */
		r_to_q (eqpk -> E.x.a, terms -> a [k].x);
		r_to_q (eqpk -> E.y.a, terms -> a [k].y);
		mpq_set_ui (eqpk -> E.x.b, 0, 1);
		mpq_set_ui (eqpk -> E.y.b, 0, 1);

		eqpk -> L = NULL;
		eqpk -> R = NULL;
	}

	/* Initialize the eq-point for each Steiner point. */
	for (k = 0; k < ns; k++) {
		eqpk = &(gcp -> eqpts [nt + k]);

		eqpk -> index	= nt + k;
		eqpk -> term	= FALSE;

		eqpk -> Efp = steins -> a [k];
		eqpk -> L = NULL;
		eqpk -> R = NULL;
	}
	/* Create adjacency lists for each vertex. */
	n = nt + ns;
	for (j = 0; j < n; j++) {
		count [j] = 0;
	}
	for (j = 0; j < ne; j++) {
		ep = &edges [j];
		++(count [ep -> p1]);
		++(count [ep -> p2]);
	}
	k = 0;
	for (j = 0; j < n; j++) {
		start [j] = k;
		ptrs [j] = k;
		k += count [j];
	}
	start [j] = k;
	for (j = 0; j < ne; j++) {
		ep = &edges [j];

		k = (ptrs [ep -> p1])++;
		adj_edge [k] = j;
		adj_vert [k] = ep -> p2;

		k = (ptrs [ep -> p2])++;
		adj_edge [k] = j;
		adj_vert [k] = ep -> p1;
	}

	/* There should be a unique edge referring to vertex 0 (a terminal). */
	v1 = 0;
	FATAL_ERROR_IF (count [v1] NE 1);

	/* Get the edge index and other vertex v2: edge e1 = (v1,v2). */
	k = start [v1];
	e1 = adj_edge [k];
	v2 = adj_vert [k];

	/* Link the eq-points together into the correct	recursive structure. */
	/* Also computes exact coorindates for each eq-point, and for the    */
	/* eq-point circle centers.					     */
	eqpk = traverse_fst_graph (gcp, fsp, v1, e1, v2);

	/* Compute FST length**2 in exact Q(sqrt(3)).  The Simpson	*/
	/* line goes from eqpts [v1] to eqpts [v2].  Store this in	*/
	/*gcp -> fst_qr3 [i].						*/

	compute_qr3_length_squared (&(gcp -> fst_qr3 [i]),
				    &(gcp -> eqpts [v1].E),
				    &(gcp -> eqpts [v2].E));

	/* Second traversal to compute Steiner point locations in	*/
	/* exact Q(sqrt(3)).						*/
	traverse_compute_exact_steiner_points (gcp,
					       fsp,
					       &(gcp -> eqpts [v1].E),
					       &(gcp -> eqpts [v2]));

#if 0
	for (j = 0; j < ns; j++) {
		printf ("---------------------\n"
			" FST %d / Steiner %d:\n", i, j);
		printf (" From EFST:\n"
			"    X = %.17g\n"
			"    Y = %.17g\n",
			fsp -> steiners -> a [j].x,
			fsp -> steiners -> a [j].y);
		printf (" From Q(sqrt(3)):\n"
			"    X = ");
		print_qr3 (&(gcp -> eqpts [nt + j].STP.x));
		printf ("\n");
		printf ("    Y = ");
		print_qr3 (&(gcp -> eqpts [nt + j].STP.y));
		printf ("\n");
		printf (" From MPF:\n");
		qr3_t rtemp;
		mpf_t ftemp;
		mpf_init2 (ftemp, gcp -> fp_numbits);
		_gst_qr3_init (&rtemp);
		qr3_mul (&rtemp,
			 &(gcp -> eqpts [nt + j].STP.x),
			 &(gcp -> eqpts [nt + j].STP.x));
		printf ("    X (double) = %.17g\n",
			sqrt (mpq_get_d (rtemp.a) + mpq_get_d (rtemp.b) * sqrt (3.0)));
		compute_sqrt_scaled_qr3 (ftemp, &rtemp, NULL);
		printf ("    X = %.17g\n", mpf_get_d (ftemp));
		qr3_mul (&rtemp,
			 &(gcp -> eqpts [nt + j].STP.y),
			 &(gcp -> eqpts [nt + j].STP.y));
		printf ("    Y (double) = %.17g\n",
			sqrt (mpq_get_d (rtemp.a) + mpq_get_d (rtemp.b) * sqrt (3.0)));
		compute_sqrt_scaled_qr3 (ftemp, &rtemp, NULL);
		printf ("    Y = %.17g\n", mpf_get_d (ftemp));
		_gst_qr3_clear (&rtemp);
		mpf_clear (ftemp);
	}
#endif

	k = gcp -> edge_base [i];
	if (k < 0) {
		/* This FST is not being retained.			*/
		/* No place to store the edge lengths for this FST.	*/
		return;
	}

	/* Compute exact edge length*2 values in Q(sqrt(3)).		*/
	/* Store these in gcp -> edge_qr3[k], where k starts at		*/
	/* gcp -> edge_base [i].					*/

	for (j = 0; j < ne; j++) {
		ep = &edges [j];

		if (ep -> p1 < nt) {
			qp1 = &(gcp -> eqpts [ep -> p1].E);
		}
		else {
			qp1 = &(gcp -> eqpts [ep -> p1].STP);
		}

		if (ep -> p2 < nt) {
			qp2 = &(gcp -> eqpts [ep -> p2].E);
		}
		else {
			qp2 = &(gcp -> eqpts [ep -> p2].STP);
		}

		compute_qr3_length_squared (&(gcp -> edge_qr3 [k + j]), qp1, qp2);
	}
}

/*
 * Traverse the FST graph structure recursively, translating this
 * into properly formed eq-point tree.  In other words, we need to
 * link the various eq-point objects together to properly mimic the
 * geometric construction of the equilateral-point corresponding
 * to the subtree rooted at vertex v1 emanating down through
 * edge e1, to vertex v2.
 *
 * The only really tricky part of this is that when v2 is a Steiner
 * point, it branches to two other eq-points, and we must determine
 * the correct geometric ordering of these two eq-points.  We do
 * this using floating-point arithmetic, which hopefully will not
 * make the wrong choice.  Given eq-points P and Q, we choose
 * (Q,P) if (v2-P)X(Q-P) > 0, and (P,Q) otherwise.  Because the
 * angles should be 120 degrees, the floating-point arithmetic should
 * not misbehave much.
 *
 * Returns the eq-point corresponding to the subtree e1==>v2.
 */

	static
	struct ExEqp *
traverse_fst_graph (

struct GceInfo *	gcp,		/* IN/OUT: global data */
struct full_set *	fsp,		/* IN: FST being traversed */
int			v1,		/* IN: vertex v1 */
int			e1,		/* IN: edge index */
int			v2		/* IN: vertex v2 */
)
{
int			j, k, k1, k2, nt;
int			e2, v3;
struct ExEqp *		p;
struct ExEqp *		q;
struct ExEqp *		r;
int			adjbuf [2];
struct qr3_point	R;
mpq_t			c2, c3, t1, t2;

	nt = fsp -> terminals -> n;

	if (v2 < nt) {
		/* A terminal.  We are done! */
		return (&(gcp -> eqpts [v2]));
	}

	k1 = gcp -> start [v2];
	k2 = gcp -> start [v2 + 1];
	j = 0;
	for (k = k1; k < k2; k++) {
		e2 = gcp -> adj_edge [k];
		if (e2 EQ e1) continue;
		adjbuf [j++] = k;
	}

	FATAL_ERROR_IF (j NE 2);

	k = adjbuf [0];
	e2 = gcp -> adj_edge [k];
	v3 = gcp -> adj_vert [k];
	p = traverse_fst_graph (gcp, fsp, v2, e2, v3);

	k = adjbuf [1];
	v3 = gcp -> adj_vert [k];
	e2 = gcp -> adj_edge [k];
	q = traverse_fst_graph (gcp, fsp, v2, e2, v3);

#if 1
	/* Our FST generator always puts P and then Q. */
#else
	/* Decide which order to put P and Q in (swap if necessary). */
	r = &(gcp -> eqpts [v2]);
	{
		double cprod;
		cprod = (r -> Efp.x - p -> Efp.x) * (q -> Efp.y - p -> Efp.y)
			- (r -> Efp.y - p -> Efp.y) * (q -> Efp.x - p -> Efp.x);
		if (cprod < 0) {
			r = q;
			q = p;
			p = r;
		}
	}
#endif

	r = &(gcp -> eqpts [v2]);
	r -> L	= q;
	r -> R	= p;

	/* Compute exact Q(sqrt(3)) coordinates of this eq-point. */

	_gst_qr3_init (&R.x);
	_gst_qr3_init (&R.y);

	mpq_sub (R.x.a, q -> E.x.a, p -> E.x.a);
	mpq_sub (R.x.b, q -> E.x.b, p -> E.x.b);
	mpq_sub (R.y.a, q -> E.y.a, p -> E.y.a);
	mpq_sub (R.y.b, q -> E.y.b, p -> E.y.b);

	mpz_init_set_ui (mpq_numref (c2), 2);
	mpz_init_set_ui (mpq_denref (c2), 1);

	mpz_init_set_ui (mpq_numref (c3), 3);
	mpz_init_set_ui (mpq_denref (c3), 1);

	mpq_init (t1);
	mpq_init (t2);

	mpq_mul (t1, c3, R.y.b);
	mpq_sub (t2, R.x.a, t1);
	mpq_div (t1, t2, c2);
	mpq_add (r -> E.x.a, p -> E.x.a, t1);

	mpq_sub (t1, R.x.b, R.y.a);
	mpq_div (t2, t1, c2);
	mpq_add (r -> E.x.b, p -> E.x.b, t2);

	mpq_mul (t1, c3, R.x.b);
	mpq_add (t2, R.y.a, t1);
	mpq_div (t1, t2, c2);
	mpq_add (r -> E.y.a, p -> E.y.a, t1);

	mpq_add (t1, R.y.b, R.x.a);
	mpq_div (t2, t1, c2);
	mpq_add (r -> E.y.b, p -> E.y.b, t2);

	/* Compute exact coordinates of eq-point circle center. */

	mpq_add (r -> DC.x.a, p -> E.x.a, q -> E.x.a);
	mpq_add (r -> DC.x.b, p -> E.x.b, q -> E.x.b);
	mpq_add (r -> DC.y.a, p -> E.y.a, q -> E.y.a);
	mpq_add (r -> DC.y.b, p -> E.y.b, q -> E.y.b);
	mpq_add (r -> DC.x.a, r -> DC.x.a, r -> E.x.a);
	mpq_add (r -> DC.x.b, r -> DC.x.b, r -> E.x.b);
	mpq_add (r -> DC.y.a, r -> DC.y.a, r -> E.y.a);
	mpq_add (r -> DC.y.b, r -> DC.y.b, r -> E.y.b);
	mpq_div (r -> DC.x.a, r -> DC.x.a, c3);
	mpq_div (r -> DC.x.b, r -> DC.x.b, c3);
	mpq_div (r -> DC.y.a, r -> DC.y.a, c3);
	mpq_div (r -> DC.y.b, r -> DC.y.b, c3);

	mpq_clear (t2);
	mpq_clear (t1);
	mpq_clear (c3);
	mpq_clear (c2);

	_gst_qr3_clear (&R.y);
	_gst_qr3_clear (&R.x);

	return (r);
}

/*
 * Compute the exact length**2 of the line segment between points P1 and P2,
 * both of whose coordinates are in Q(sqrt(3)).  The result is exact
 * and also resides in Q(sqrt(3)).
 */

	static
	void
compute_qr3_length_squared (

qr3_t *			result,		/* IN/OUT: resulting length**2 */
struct qr3_point *	p1,		/* IN: point P1 */
struct qr3_point *	p2		/* IN: point P2 */
)
{
struct qr3_point	DELTA;
qr3_t			A, B;

	_gst_qr3_init (&DELTA.x);
	_gst_qr3_init (&DELTA.y);
	_gst_qr3_init (&A);
	_gst_qr3_init (&B);

	mpq_sub (DELTA.x.a, p1 -> x.a, p2 -> x.a);
	mpq_sub (DELTA.x.b, p1 -> x.b, p2 -> x.b);
	mpq_sub (DELTA.y.a, p1 -> y.a, p2 -> y.a);
	mpq_sub (DELTA.y.b, p1 -> y.b, p2 -> y.b);

	qr3_mul (&A, &DELTA.x, &DELTA.x);
	qr3_mul (&B, &DELTA.y, &DELTA.y);

	mpq_add (result -> a, A.a, B.a);
	mpq_add (result -> b, A.b, B.b);

	_gst_qr3_clear (&B);
	_gst_qr3_clear (&A);
	_gst_qr3_clear (&DELTA.y);
	_gst_qr3_clear (&DELTA.x);
}

/*
 * A second recursive traversal of the eq-point structure, this time
 * to compute the exact coordinates (in Q(sqrt(3)) of each Steiner
 * point in this FST.
 *
 * The Simpson line begins at given qr3_point tpnt and ends at eq-point
 * R.
 */

	static
	void
traverse_compute_exact_steiner_points (

struct GceInfo *	gcp,		/* IN/OUT: global data */
struct full_set *	fsp,		/* IN: FST being traversed */
struct qr3_point *	tpnt,		/* IN: head coords of Simpson line */
struct ExEqp *		r		/* IN: eq-point R */
)
{
	if (r -> term) {
		/* A terminal.  We are done! */
		return;
	}

	/* Compute exact Steiner point STP by projecting segment	*/
	/* from r->E to tpnt onto other side of the circle.		*/
	project_qr3_point (&(r -> STP),
			   &(r -> E),
			   &(r -> DC),
			   tpnt);

	/* Using this Steiner point as the new "root", traverse both	*/
	/* the left and right sub-trees.				*/

	traverse_compute_exact_steiner_points (gcp, fsp, &(r -> STP), r -> L);

	traverse_compute_exact_steiner_points (gcp, fsp, &(r -> STP), r -> R);
}

/*
 * Print a Q(sqrt(3)) value both symbolically and numerically,
 * for debugging purposes.
 */

	static
	void
print_qr3 (

const qr3_t *	x
)
{
	mpz_out_str (stdout, 10, mpq_numref (x -> a));
	printf ("/");
	mpz_out_str (stdout, 10, mpq_denref (x -> a));
	printf (" + ");
	mpz_out_str (stdout, 10, mpq_numref (x -> b));
	printf ("/");
	mpz_out_str (stdout, 10, mpq_denref (x -> b));
	printf (" = ");
	printf (" %.17g\n",
		mpq_get_d (x -> a) + mpq_get_d (x -> b) * sqrt (3.0));
}

/*
 * Print a Q(sqrt(3)) point, both symbolically and numerically
 * for debugging purposes.
 */

	void
_gst_print_qr3_point (

const struct qr3_point *	pnt
)
{
	printf ("X = ");
	print_qr3 (&(pnt -> x));
	printf ("Y = ");
	print_qr3 (&(pnt -> y));
}

/*
 * Given a point E on a circle with center C, return the projection S
 * of E onto the same circle in the direction of the ray from E through T.
 *
 * All point coordinates are in exact Q(sqrt(3)) representation.
 */

	static
	void
project_qr3_point (

struct qr3_point *		S,	/* OUT: resulting point S */
const struct qr3_point *	E,	/* IN: point E to project */
const struct qr3_point *	C,	/* IN: center of circle */
const struct qr3_point *	T	/* IN: direction of projection */
)
{
struct qr3_point	A, B;
qr3_t			AdotB, BdotB, lambda;

	_gst_qr3_init (&A.x);
	_gst_qr3_init (&A.y);
	_gst_qr3_init (&B.x);
	_gst_qr3_init (&B.y);
	_gst_qr3_init (&AdotB);	
	_gst_qr3_init (&BdotB);
	_gst_qr3_init (&lambda);

	/* A = C - E; */
	mpq_sub (A.x.a, C -> x.a, E -> x.a);
	mpq_sub (A.x.b, C -> x.b, E -> x.b);
	mpq_sub (A.y.a, C -> y.a, E -> y.a);
	mpq_sub (A.y.b, C -> y.b, E -> y.b);

	/* B = T - E; */
	mpq_sub (B.x.a, T -> x.a, E -> x.a);
	mpq_sub (B.x.b, T -> x.b, E -> x.b);
	mpq_sub (B.y.a, T -> y.a, E -> y.a);
	mpq_sub (B.y.b, T -> y.b, E -> y.b);

	qr3_dot_product (&AdotB, &A, &B);
	qr3_dot_product (&BdotB, &B, &B);

	if ((mpq_sgn (BdotB.a) EQ 0) AND (mpq_sgn (BdotB.b) EQ 0)) {
		mpq_set (S -> x.a, E -> x.a);
		mpq_set (S -> x.b, E -> x.b);
		mpq_set (S -> y.a, E -> y.a);
		mpq_set (S -> y.b, E -> y.b);
	}
	else {
		/* lambda = AdotB / BdotB; */
		qr3_div (&lambda, &AdotB, &BdotB);

		/* lambda = 2 * lambda; */
		mpq_add (lambda.a, lambda.a, lambda.a);
		mpq_add (lambda.b, lambda.b, lambda.b);

		/* TEMP = lambda * B; */
		qr3_mul (&AdotB, &lambda, &B.x);
		qr3_mul (&BdotB, &lambda, &B.y);

		/* S = E + TEMP; */
		mpq_add (S -> x.a, E -> x.a, AdotB.a);
		mpq_add (S -> x.b, E -> x.b, AdotB.b);
		mpq_add (S -> y.a, E -> y.a, BdotB.a);
		mpq_add (S -> y.b, E -> y.b, BdotB.b);
	}

	_gst_qr3_clear (&lambda);
	_gst_qr3_clear (&BdotB);
	_gst_qr3_clear (&AdotB);
	_gst_qr3_clear (&B.y);
	_gst_qr3_clear (&B.x);
	_gst_qr3_clear (&A.y);
	_gst_qr3_clear (&A.x);
}

/*
 * Compute the dot-product of two points A and B using
 * exact Q(sqrt(3)) arithmetic.
 */

	static
	void
qr3_dot_product (

qr3_t *				RES,	/* OUT: result */
const struct qr3_point *	A,	/* IN: first point A */
const struct qr3_point *	B	/* IN: second point B */
)
{
qr3_t		PRODX, PRODY;

	_gst_qr3_init (&PRODX);
	_gst_qr3_init (&PRODY);

	qr3_mul (&PRODX, &(A -> x), &(B -> x));
	qr3_mul (&PRODY, &(A -> y), &(B -> y));

	mpq_add (RES -> a, PRODX.a, PRODY.a);
	mpq_add (RES -> b, PRODX.b, PRODY.b);

	_gst_qr3_clear (&PRODY);
	_gst_qr3_clear (&PRODX);
}

/*
 * Perform division using exact Q(sqrt(3)) arithmetic.
 */

	static
	void
qr3_div (

qr3_t *		RES,			/* OUT: resulting quotient */
const qr3_t *	A,			/* IN: dividend A */
const qr3_t *	B			/* IN: divisor B */
)
{
mpq_t		t1, t2, t3, c3, denom;

	mpq_init (t1);
	mpq_init (t2);
	mpq_init (t3);
	mpq_init (c3);
	mpq_init (denom);

	mpq_set_ui (c3, 3, 1);

	/* denom = 3 * Bb**2 - Ba**2; */
	mpq_mul (t1, B -> b, B -> b);
	mpq_mul (t1, t1, c3);
	mpq_mul (t2, B -> a, B -> a);
	mpq_sub (denom, t1, t2);

	/* t1 = 3*Ab*Bb - Aa*Ba; */
	mpq_mul (t1, A -> b, B -> b);
	mpq_mul (t1, t1, c3);
	mpq_mul (t2, A -> a, B -> a);
	mpq_sub (t1, t1, t2);

	/* t2 = Aa*Bb - Ab*Ba; */
	mpq_mul (t2, A -> a, B -> b);
	mpq_mul (t3, A -> b, B -> a);
	mpq_sub (t2, t2, t3);

	/* RES = (t1 / denom) + (t2 / denom)*sqrt(3); */
	mpq_div (RES -> a, t1, denom);
	mpq_div (RES -> b, t2, denom);

	mpq_clear (denom);
	mpq_clear (c3);
	mpq_clear (t3);
	mpq_clear (t2);
	mpq_clear (t1);
}

/*
 * Compute the square-root of the given exact element of Q(sqrt(3)),
 * scaled by a given rational factor (which can be NULL, meaning 1).
 * The result is in high-precision floating-point form, and the
 * working precision is taken as an input from this result value.
 *
 * We are given A = (a + b*sqrt(3)).  We seek a Z^2 = Y such that:
 *
 *		Y = A = a + b*sqrt(3)
 * ==>		Y - a = b*sqrt(3)
 * ==>		(Y - a)**2 = 3 * b**2
 * ==>		Y**2 - 2*a*Y + a**2 = 3 * b**2
 * ==>		Y**2 - 2*a*Y + a**2 - 3*b**2 = 0
 *
 * ==>		Z**4 - 2*a*Z**2 + a**2 - 3*b**2 = 0
 *
 * Newton iteration on this polynomial yields:
 *
 *	Z_new = Z - (Z**4 - 2*a*Z**2 - 3*b**2 + a**2) / (4*Z**3 - 4*a*Z)
 *
 * We start with a direct floating-point approximation.
 */

	static
	void
compute_sqrt_scaled_qr3 (

mpf_ptr		res,		/* IN/OUT: resulting high-precision value */
const qr3_t *	A,		/* IN: exact element of Q(sqrt(3)) */
mpq_srcptr	scale		/* IN: scale factor (or NULL) */
)
{
mpq_t		a, b;

	mpq_init (a);
	mpq_init (b);

	/* Scale a local copy of the given argument. */
	mpq_set (a, A -> a);
	mpq_set (b, A -> b);
	if (scale NE NULL) {
		mpq_mul (a, a, scale);
		mpq_mul (b, b, scale);
	}

	/* Newton loses its quadratic convergence if you are trying to	*/
	/* approximate a root of multiplicity greater than 1!  Check	*/
	/* for the simple cases of this first, a EQ 0 or b EQ 0.	*/

	if (mpz_sgn (mpq_numref (a)) EQ 0) {
		if (mpz_sgn (mpq_numref (b)) EQ 0) {
			/* (a EQ 0) AND (b EQ 0).  Result is zero. */
			mpf_set_ui (res, 0);
		}
		else {
			/* (a EQ 0) AND (b NE 0). */
			compute_sqrt_b_root3 (res, b);
		}
	}
	else if (mpz_sgn (mpq_numref (b)) EQ 0) {
		/* (a NE 0) AND (b EQ 0). */
		compute_sqrt_a (res, a);
	}
	else {
		/* (a NE 0) AND (b NE 0). */
		compute_sqrt_a_plus_b_root3 (res, a, b);
	}

	mpq_clear (b);
	mpq_clear (a);
}

/*
 * Compute sqrt (a), where a is a rational number N/D.  This corresponds
 * to finding the root of D**2 * Z**2 - N**2 = 0.
 */

	static
	void
compute_sqrt_a (

mpf_ptr		res,		/* IN/OUT: resulting high-precision value */
mpq_srcptr	a		/* IN: rational number a */
)
{
int		precision;
bool		converged;
mpz_srcptr	N, D;
double		zfp;
mpf_t		z, t1, t2, fn, fd;
mpf_t		num, den, mag;

	precision = mpf_get_prec (res);

	N = mpq_numref (a);
	D = mpq_denref (a);

	if (mpz_perfect_square_p (N) AND mpz_perfect_square_p (D)) {
		mpq_t		TMP;
		mpq_init (TMP);
		mpz_sqrt (mpq_numref (TMP), N);
		mpz_sqrt (mpq_denref (TMP), D);
		mpq_canonicalize (TMP);
		mpf_set_q (res, TMP);
		mpq_clear (TMP);
		return;
	}

	/* Use Newton Z' = Z - (D*Z^2 - N) / (2 * D). */
	zfp = mpq_get_d (a);
	zfp = sqrt (zfp);

	mpf_init2 (z, precision);
	mpf_init2 (t1, precision);
	mpf_init2 (t2, precision);
	mpf_init2 (fn, precision);
	mpf_init2 (fd, precision);
	mpf_init2 (num, precision);
	mpf_init2 (den, precision);
	mpf_init2 (mag, precision);

	mpf_set_z (fn, N);
	mpf_set_z (fd, D);

	mpf_set_d (z, zfp);

	for (;;) {
		mpf_mul (num, z, z);
		mpf_mul (num, num, fd);
		mpf_sub (num, num, fn);
		mpf_mul (den, fd, z);
		mpf_mul_ui (den, den, 2);
		mpf_div (t1, num, den);

		/* Check tolerance. */
		mpf_div_2exp (mag, z, precision);
		mpf_sub (mag, mag, t1);
		mpf_abs (mag, mag);
		mpf_mul_2exp (mag, mag, precision);
		mpf_div (mag, mag, z);
		converged = (mpf_cmp_ui (mag, 256) < 0);

		mpf_sub (z, z, t1);

		if (converged) break;
	}

	mpf_set (res, z);

	mpf_clear (mag);
	mpf_clear (den);
	mpf_clear (num);
	mpf_clear (fd);
	mpf_clear (fn);
	mpf_clear (t2);
	mpf_clear (t1);
	mpf_clear (z);
}

/*
 * Compute sqrt (b * sqrt(3)), where b is a rational number N/D.
 *  This corresponds to finding the root of D**2 * Z**4 - 3* N**2 = 0.
 */

	static
	void
compute_sqrt_b_root3 (

mpf_ptr		res,		/* IN/OUT: resulting high-precision value */
mpq_srcptr	b		/* IN: rational number b */
)
{
int		precision;
mpz_srcptr	N, D;
double		zfp;
mpf_t		z, z2, z4, t1, t2, f1, f2, f3;
mpz_t		TMP;

	precision = mpf_get_prec (res);

	N = mpq_numref (b);
	D = mpq_denref (b);

	/* Use Newton Z' = Z - (D^2*Z^4 - 3*N^2) / (4*D^2*Z^3). */
	zfp = mpq_get_d (b);
	zfp *= sqrt (3.0);
	zfp = sqrt (zfp);

	mpz_init (TMP);

	mpf_init2 (z, precision);
	mpf_init2 (z2, precision);
	mpf_init2 (z4, precision);
	mpf_init2 (t1, precision);
	mpf_init2 (t2, precision);
	mpf_init2 (f1, precision);
	mpf_init2 (f2, precision);
	mpf_init2 (f3, precision);

	mpz_mul (TMP, D, D);
	mpf_set_z (f1, TMP);

	mpz_mul (TMP, N, N);
	mpz_mul_si (TMP, TMP, -3);
	mpf_set_z (f2, TMP);

	mpz_mul (TMP, D, D);
	mpz_mul_ui (TMP, TMP, 4);
	mpf_set_z (f3, TMP);

	mpf_set_d (z, zfp);

	for (;;) {
		mpf_mul (z2, z, z);
		mpf_mul (z4, z2, z2);

		mpf_mul (t1, f1, z4);
		mpf_add (t1, t1, f2);

		mpf_mul (t2, f3, z2);
		mpf_mul (t2, t2, z);

		mpf_div (t1, t1, t2);
		mpf_sub (t2, z, t1);

		if (mpf_cmp (t2, z) EQ 0) break;
		mpf_set (z, t2);
	}

	mpf_set (res, z);

	mpf_clear (f3);
	mpf_clear (f2);
	mpf_clear (f1);
	mpf_clear (t2);
	mpf_clear (t1);
	mpf_clear (z4);
	mpf_clear (z2);
	mpf_clear (z);

	mpz_clear (TMP);
}

/*
 * Compute sqrt (a + b * sqrt(3)), where a and b are both non-zero rational
 * numbers.
 */

	static
	void
compute_sqrt_a_plus_b_root3 (

mpf_ptr		res,		/* IN/OUT: resulting high-precision value */
mpq_srcptr	a,		/* IN: rational number a */
mpq_srcptr	b		/* IN: rational number b */
)
{
int		precision;
bool		converged;
double		zfp;
mpq_t		a2, b2;
mpq_t		c2, c3, c4;
mpq_t		qn4, qn2, qn0, qd3, qd1;
mpz_t		d, g;
mpz_ptr		zn4, zn2, zn0, zd3, zd1;
mpf_t		fn4, fn2, fn0, fd3, fd1;
mpf_t		z, z2, num, den, mag;

	precision = mpf_get_prec (res);

	mpq_init (a2);
	mpq_init (b2);
	mpq_init (c2);
	mpq_init (c3);
	mpq_init (c4);
	mpq_init (qn4);
	mpq_init (qn2);
	mpq_init (qn0);
	mpq_init (qd3);
	mpq_init (qd1);

	mpz_init (d);
	mpz_init (g);

	mpf_init2 (fn4, precision);
	mpf_init2 (fn2, precision);
	mpf_init2 (fn0, precision);
	mpf_init2 (fd3, precision);
	mpf_init2 (fd1, precision);
	mpf_init2 (z, precision);
	mpf_init2 (z2, precision);
	mpf_init2 (num, precision);
	mpf_init2 (den, precision);
	mpf_init2 (mag, precision);

#if 0
	printf ("sqrt(%.17g + %.17g*sqrt(3)) = %.17g\n",
		mpq_get_d (a),
		mpq_get_d (b),
		sqrt (mpq_get_d (a) + mpq_get_d (b) * sqrt (3.0)));
	printf ("a = ");
	mpz_out_str (stdout, 10, mpq_numref (a));
	printf (" / ");
	mpz_out_str (stdout, 10, mpq_denref (a));
	printf ("\n");
	printf ("b = ");
	mpz_out_str (stdout, 10, mpq_numref (b));
	printf (" / ");
	mpz_out_str (stdout, 10, mpq_denref (b));
	printf ("\n");
#endif

	mpq_set_ui (c2, 2, 1);
	mpq_set_ui (c3, 3, 1);
	mpq_set_ui (c4, 4, 1);

	mpq_mul (a2, a, a);
	mpq_mul (b2, b, b);

	/* Compute the exact rational coefficients (qn4, qn2, qn0) for	*/
	/* the numerator polynomial.					*/
	mpq_set_ui (qn4, 1, 1);
	mpq_mul (qn2, a, c2);
	mpq_neg (qn2, qn2);
	mpq_mul (qn0, c3, b2);
	mpq_sub (qn0, a2, qn0);

	/* Compute the exact rational coefficients (qd3, qd1) for the	*/
	/* denominator polynomial.					*/
	mpq_set (qd3, c4);
	mpq_mul (qd1, c4, a);
	mpq_neg (qd1, qd1);

	/* Compute the least-common-multiple of the denominators of	*/
	/* the coefficients in both polynomials.			*/

#define UPDATE_LCM(coeff)				\
	mpz_gcd (g, d, mpq_denref (coeff));		\
	mpz_divexact (g, mpq_denref (coeff), g);	\
	mpz_mul (d, d, g)

	mpz_set_ui (d, 1);
	UPDATE_LCM (qn4);
	UPDATE_LCM (qn2);
	UPDATE_LCM (qn0);
	UPDATE_LCM (qd3);
	UPDATE_LCM (qd1);

#undef UPDATE_LCM

	/* Now clear the denominators in these coefficients. */

#define CLEAR_DENOM(d, coeff)					\
	mpz_mul (mpq_numref (coeff), mpq_numref (coeff), d);	\
	mpq_canonicalize (coeff);				\
	if (mpz_cmp_ui (mpq_denref (coeff), 1) NE 0) {		\
		FATAL_ERROR;					\
	}

	CLEAR_DENOM (d, qn4);
	CLEAR_DENOM (d, qn2);
	CLEAR_DENOM (d, qn0);
	CLEAR_DENOM (d, qd3);
	CLEAR_DENOM (d, qd1);

	/* Get pointers to the numerator coefficients that remain. */
	zn4 = mpq_numref (qn4);
	zn2 = mpq_numref (qn2);
	zn0 = mpq_numref (qn0);
	zd3 = mpq_numref (qd3);
	zd1 = mpq_numref (qd1);

	/* Find common factors in the numerator. */
	mpz_gcd (g, zn4, zn2);
	mpz_gcd (g, g, zn0);

	/* Find common factors in the denominator. */
	mpz_gcd (d, zd3, zd1);

	/* See if there is any factor common to both that can be canceled. */
	mpz_gcd (g, g, d);
	if (mpz_cmp_ui (g, 1) NE 0) {
		mpz_divexact (zn4, zn4, g);
		mpz_divexact (zn2, zn2, g);
		mpz_divexact (zn0, zn0, g);
		mpz_divexact (zd3, zd3, g);
		mpz_divexact (zd1, zd1, g);
	}

	/* Place integer coefficients into floating-point form. */
	mpf_set_z (fn4, zn4);
	mpf_set_z (fn2, zn2);
	mpf_set_z (fn0, zn0);
	mpf_set_z (fd3, zd3);
	mpf_set_z (fd1, zd1);

#if 0
	printf ("Polynomial Coefficients:\n");
#define FOO(coeff, var, punct) \
	printf ("  " #coeff " = "); \
	mpz_out_str (stdout, 10, var); \
	printf ("%s\n", punct);

	printf ("vals : [\n");
	FOO (qn4, zn4, ",")
	FOO (qn2, zn2, ",")
	FOO (qn0, zn0, ",")
	FOO (qd3, zd3, ",")
	FOO (qd1, zd1, "")
	printf ("]\n");
#endif

	/* Compute floating-point approximation to start with. */
#if 0
	mpf_set_q (num, a);
	mpf_set_q (den, b);
	mpf_set_ui (z2, 3);
	mpf_sqrt (z2, z2);
	mpf_mul (z2, z2, den);
	mpf_add (z, num, z2);
#else
	zfp = mpq_get_d (a) + mpq_get_d (b) * sqrt (3.0);
	zfp = sqrt (zfp);
	mpf_set_d (z, zfp);
#endif

#if 0
	printf ("  Initial estimate: %.17g\n", zfp);
#endif

	/* The Newton iteration. */
	for (;;) {
		mpf_mul (z2, z, z);

		mpf_mul (num, fn4, z2);
		mpf_add (num, num, fn2);
		mpf_mul (num, num, z2);
		mpf_add (num, num, fn0);

		mpf_mul (den, fd3, z2);
		mpf_add (den, den, fd1);
		mpf_mul (den, den, z);

		mpf_div (num, num, den);

		/* Check tolerance. */
		mpf_div_2exp (mag, z, precision);
		mpf_sub (mag, mag, num);
		mpf_abs (mag, mag);
		mpf_mul_2exp (mag, mag, precision);
		mpf_div (mag, mag, z);
		converged = (mpf_cmp_ui (mag, 256) < 0);

		mpf_sub (z, z, num);

#if 0
		printf ("  New iterate: %.17g\n", mpf_get_d (z));
#endif

		if (converged) break;
	}

#if 0
	printf ("  Final result: %.17g\n", mpf_get_d (z));
#endif

	mpf_set (res, z);

	mpf_clear (mag);
	mpf_clear (den);
	mpf_clear (num);
	mpf_clear (z2);
	mpf_clear (z);
	mpf_clear (fd1);
	mpf_clear (fd3);
	mpf_clear (fn0);
	mpf_clear (fn2);
	mpf_clear (fn4);

	mpz_clear (g);
	mpz_clear (d);

	mpq_clear (qd1);
	mpq_clear (qd3);
	mpq_clear (qn0);
	mpq_clear (qn2);
	mpq_clear (qn4);
	mpq_clear (c4);
	mpq_clear (c3);
	mpq_clear (c2);
	mpq_clear (b2);
	mpq_clear (a2);
}

/*
 * Let K = num_bits.
 * Compute the largest rational scale factor of the form 10**m (where m
 * can be positive, zero or negative) such that:
 *
 *	(9/8) * 10**m * MST_length < 2**K
 *
 * This is a scaling factor that causes the entire Steiner problem in
 * graph instance to have edge weights that fit within K-bit integers.
 * All reasonable solution vectors should have total lengts that do
 * not overflow within unsigned K-bit integer arithmetic.
 */

	static
	void
compute_scale_factor (

mpq_ptr			scale,		/* IN/OUT: rational scale factor */
struct GceInfo *	gcp		/* IN/OUT: global data */
)
{
int			i, k, n, num_bits, fp_numbits;
struct full_set **	full_trees;
mpf_t			mstlen, edgelen, scale_target, tmp;
mpz_t			powten;

	num_bits	= gcp -> num_bits;
	fp_numbits	= gcp -> fp_numbits;
	full_trees	= gcp -> cip -> full_trees;

	mpf_init2 (mstlen, fp_numbits);
	mpf_init2 (edgelen, fp_numbits);
	mpf_init2 (scale_target, fp_numbits);
	mpf_init2 (tmp, fp_numbits);

	mpz_init (powten);

	/* Compute the scale target, which is 2**num_bits.  This should	*/
	/* always have an exact representation as mpf_t.		*/
	mpf_set_ui (scale_target, 2);
	mpf_pow_ui (scale_target, scale_target, num_bits);

	/* Compute the MST length in high-precision floating-point.	*/
	/* We assume that all 2-edges form an MST!			*/

	n = gcp -> num_fsts;
	for (i = 0; i < n; i++) {
		if (full_trees [i] -> terminals -> n EQ 2) {
			compute_sqrt_scaled_qr3 (edgelen,
						 &(gcp -> fst_qr3 [i]),
						 NULL);
			mpf_add (mstlen, mstlen, edgelen);
		}
	}

	/* Augment the length of the MST slightly. */
	mpf_mul_ui (mstlen, mstlen, 9);
	mpf_div_ui (mstlen, mstlen, 8);

	mpq_set_ui (scale, 1, 1);
	mpz_set_ui (powten, 10);

	if (mpf_cmp (mstlen, scale_target) < 0) {
		for (k = 1; ; k++) {
			mpf_set_z (tmp, powten);
			mpf_mul (tmp, tmp, mstlen);
			if (mpf_cmp (tmp, scale_target) >= 0) break;
			mpz_mul_ui (powten, powten, 10);
		}
		--k;
		mpz_ui_pow_ui (mpq_numref (scale), 10, k);
		gcp -> scale_pow = k;
	}
	else {
		for (k = 1; ; k++) {
			mpf_set_z (tmp, powten);
			mpf_div (tmp, mstlen, tmp);
			if (mpf_cmp (tmp, scale_target) < 0) break;
			mpz_mul_ui (powten, powten, 10);
		}
		mpz_ui_pow_ui (mpq_denref (scale), 10, k);
		gcp -> scale_pow = -k;
	}

	mpz_clear (powten);

	mpf_clear (tmp);
	mpf_clear (scale_target);
	mpf_clear (edgelen);
	mpf_clear (mstlen);
}

/*
 * Compute the properly scaled and rounded integer edge weights
 * for a single FST whose index is given.
 */

	static
	void
compute_edges_for_fst (

struct GceInfo *	gcp,		/* IN/OUT: global data */
mpq_srcptr		scale_factor,	/* IN: rational scale factor */
int			fstidx		/* IN: index of FST to process */
)
{
int			i, j, k, n, ne, base, fp_numbits;
struct gst_hypergraph *	cip;
struct full_set *	fsp;
struct pset *		terms;
struct pset *		steins;
struct point *		p1;
struct point *		p2;
mpf_ptr			mpf_edgelen;
mpf_ptr			mpf_edge_err;
mpz_ptr			mpz_edgelen;
double			drelerr;
mpf_t			mpf_fstlen, ftemp, mpf_one_half, relerr;
mpf_t			scale_mul, scale_div;
mpz_t			int_fstlen, total_edgelen, delta;

	cip		= gcp -> cip;
	fp_numbits	= gcp -> fp_numbits;
	base		= gcp -> edge_base [fstidx];

	fsp	= cip -> full_trees [fstidx];

	ne	= fsp -> nedges;
	terms	= fsp -> terminals;
	steins	= fsp -> steiners;

	mpf_init2 (mpf_fstlen, fp_numbits);
	mpf_init2 (ftemp, fp_numbits);
	mpf_init2 (mpf_one_half, fp_numbits);
	mpf_init2 (relerr, fp_numbits);
	mpf_init2 (scale_mul, fp_numbits);
	mpf_init2 (scale_div, fp_numbits);
	mpz_init (int_fstlen);
	mpz_init (total_edgelen);
	mpz_init (delta);

	mpf_set_ui (mpf_one_half, 1);
	mpf_div_ui (mpf_one_half, mpf_one_half, 2);

	/* Exact FST length**2 is in gcp -> fst_qr3 [fstidx].  Compute	*/
	/* its length using high-precision floating-point.		*/

	compute_sqrt_scaled_qr3 (mpf_fstlen,
				 &(gcp -> fst_qr3 [fstidx]),
				 scale_factor);
	mpf_set (&(gcp -> fst_mpf [fstidx]), mpf_fstlen);

	/* Round the FST length to the nearest integer. */
	mpf_add (ftemp, mpf_fstlen, mpf_one_half);
	mpf_floor (ftemp, ftemp);
	mpz_set_f (int_fstlen, ftemp);

	/* Allocate arrays for edge info. */
	n = 2 * ne;
	mpf_edgelen = NEWA (n, __mpf_struct);
	for (i = 0; i < n; i++) {
		mpf_init2 (&mpf_edgelen [i], fp_numbits);
	}
	mpf_edge_err = mpf_edgelen + ne;

	mpz_edgelen = NEWA (ne, __mpz_struct);
	for (i = 0; i < ne; i++) {
		mpz_init (&mpz_edgelen [i]);
	}

	k = gcp -> scale_pow;
	mpf_set_ui (scale_mul, 1);
	mpf_set_ui (scale_div, 1);
	if (k < 0) {
		mpf_set_ui (scale_div, 10);
		mpf_pow_ui (scale_div, scale_div, -k);
	}
	else {
		mpf_set_ui (scale_mul, 10);
		mpf_pow_ui (scale_mul, scale_mul, k);
	}

	/* Compute length of each intra-FST edge using high-precision	*/
	/* floating-point arithmetic.  Also round each edge length to	*/
	/* the nearest integer and record the rounding error each edge	*/
	/* yields.							*/
	for (i = 0; i < ne; i++) {
		compute_sqrt_scaled_qr3 (ftemp,
					 &(gcp -> edge_qr3 [base + i]),
					 scale_factor);
		mpf_set (&mpf_edgelen [i], ftemp);

		/* Compute relative error against unscaled floating-	*/
		/* point version of edge length.			*/
		k = fsp -> edges [i].p1;
		if (k < terms -> n) {
			p1 = &(terms -> a [k]);
		}
		else {
			k -= terms -> n;
			p1 = &(steins -> a [k]);
		}
		k = fsp -> edges [i].p2;
		if (k < terms -> n) {
			p2 = &(terms -> a [k]);
		}
		else {
			k -= terms -> n;
			p2 = &(steins -> a [k]);
		}
		mpf_set_d (relerr,
			   hypot (p1 -> x - p2 -> x, p1 -> y - p2 -> y));
		mpf_mul (relerr, relerr, scale_mul);
		mpf_div (relerr, relerr, scale_div);
		mpf_sub (relerr, relerr, ftemp);
		if (mpf_sgn (ftemp) EQ 0) {
			/* Use absolute error as relative. */
			mpf_mul (relerr, relerr, scale_div);
			mpf_div (relerr, relerr, scale_mul);
		}
		else {
			mpf_div (relerr, relerr, ftemp);
		}
		mpf_abs (relerr, relerr);
		drelerr = mpf_get_d (relerr);
		if (drelerr > gcp -> max_relerr) {
			gcp -> max_relerr = drelerr;
			gcp -> max_relerr_edge = base + i;
			gcp -> max_relerr_fst = fstidx;
		}

		/* Round to nearest integer. */
		mpf_add (ftemp, ftemp, mpf_one_half);
		mpf_floor (ftemp, ftemp);

		/* Store as integer. */
		mpz_set_f (&mpz_edgelen [i], ftemp);

		/* Record rounding error. */
		mpf_sub (&mpf_edge_err [i], &mpf_edgelen [i], ftemp);
	}

	/* Add up integer edge lengths. */
	mpz_set_ui (total_edgelen, 0);
	for (i = 0; i < ne; i++) {
		if (mpz_sgn (&mpz_edgelen [i]) < 0) {
			/* Edge weight rounds to negative value! */
			FATAL_ERROR;
		}
		mpz_add (total_edgelen, total_edgelen, &mpz_edgelen [i]);
	}

	mpz_sub (delta, int_fstlen, total_edgelen);
	if (NOT mpz_fits_sint_p (delta)) {
		FATAL_ERROR;
	}
	j = mpz_get_si (delta);
	if (j < 0) {
		j = - j;
		/* Need to decrement length of j edges.  We choose	*/
		/* those having the most negative rounding error.	*/
		for (; j > 0; j--) {
			mpf_set (ftemp, &mpf_edge_err [0]);
			k = -1;
			for (i = 0; i < ne; i++) {
				/* Do not decrement past zero. */
				if ((mpf_cmp (&mpf_edge_err [i], ftemp) <= 0) AND
				    (mpz_sgn (&mpz_edgelen [i]) > 0)) {
					mpf_set (ftemp, &mpf_edge_err [i]);
					k = i;
				}
			}
			if (k < 0) {
				/* no weight could be decremented */
				FATAL_ERROR;
			}
			mpz_sub_ui (&mpz_edgelen [k], &mpz_edgelen [k], 1);
			mpf_add_ui (&mpf_edge_err [k], &mpf_edge_err [k], 2 * ne);
		}
	}
	else if (j > 0) {
		/* Need to increment length of j edges.  We choose	*/
		/* those having the most positive rounding error.	*/
		for (; j > 0; j--) {
			mpf_set (ftemp, &mpf_edge_err [0]);
			k = 0;
			for (i = 1; i < ne; i++) {
				if (mpf_cmp (&mpf_edge_err [i], ftemp) > 0) {
					mpf_set (ftemp, &mpf_edge_err [i]);
					k = i;
				}
			}
			mpz_add_ui (&mpz_edgelen [k], &mpz_edgelen [k], 1);
			mpf_sub_ui (&mpf_edge_err [k], &mpf_edge_err [k], 2 * ne);
		}
	}

	/* Verify that integer edge lengths now add up to int_fstlen. */
	mpz_set_ui (total_edgelen, 0);
	for (i = 0; i < ne; i++) {
		FATAL_ERROR_IF (mpz_sgn (&mpz_edgelen [i]) < 0);
		mpz_add (total_edgelen, total_edgelen, &mpz_edgelen [i]);
	}
	FATAL_ERROR_IF (mpz_cmp (total_edgelen, int_fstlen) NE 0);

	/* Copy integer edge lengths into output array. */
	for (i = 0; i < ne; i++) {
		mpz_set (&(gcp -> weight [base + i]), &mpz_edgelen [i]);
	}

	/* Clean up. */

	for (i = 0; i < ne; i++) {
		mpz_clear (&mpz_edgelen [i]);
	}
	free (mpz_edgelen);

	n = 2 * ne;
	for (i = 0; i < n; i++) {
		mpf_clear (&mpf_edgelen [i]);
	}
	free (mpf_edgelen);

	mpz_clear (delta);
	mpz_clear (total_edgelen);
	mpz_clear (int_fstlen);
	mpf_clear (scale_div);
	mpf_clear (scale_mul);
	mpf_clear (relerr);
	mpf_clear (mpf_one_half);
	mpf_clear (ftemp);
	mpf_clear (mpf_fstlen);
}

/*
 * Free up the memory for the various graph structure arrays.
 */

	static
	void
free_graph_arrays (

struct GceInfo *	gcp		/* IN/OUT: global data */
)
{
int		i, num_fsts, num_edges;
int		nv;
qr3_t *		fst_qr3;
qr3_t *		edge_qr3;
mpf_ptr		fst_mpf;
mpf_ptr		edge_mpf;
struct ExEqp *	eqpts;
struct ExEqp *	eqpk;

	free (gcp -> count);
	gcp -> count	= NULL;
	gcp -> start	= NULL;
	gcp -> ptrs	= NULL;
	gcp -> adj_edge	= NULL;
	gcp -> adj_vert	= NULL;

	num_fsts	= gcp -> num_fsts;
	num_edges	= gcp -> num_edges;

	nv = gcp -> max_terms + gcp -> max_steins;

	eqpts = gcp -> eqpts;
	gcp -> eqpts = NULL;
	for (i = 0; i < nv; i++) {
		eqpk = &eqpts [i];
		_gst_qr3_clear (&(eqpk -> E.x));
		_gst_qr3_clear (&(eqpk -> E.y));
		_gst_qr3_clear (&(eqpk -> DC.x));
		_gst_qr3_clear (&(eqpk -> DC.y));
		_gst_qr3_clear (&(eqpk -> STP.x));
		_gst_qr3_clear (&(eqpk -> STP.y));
	}
	free (eqpts);

	/* Do *NOT* free up gcp -> weight -- control of this array is	*/
	/* handed over to the cost_extension object.			*/
	gcp -> weight = NULL;

	edge_mpf	= gcp -> edge_mpf;
	gcp -> edge_mpf	= NULL;
	for (i = 0; i < num_edges; i++) {
		mpf_clear (&edge_mpf [i]);
	}
	free (edge_mpf);

	fst_mpf		= gcp -> fst_mpf;
	gcp -> fst_mpf	= NULL;
	for (i = 0; i < num_fsts; i++) {
		mpf_clear (&fst_mpf [i]);
	}
	free (fst_mpf);

	edge_qr3	= gcp -> edge_qr3;
	gcp -> edge_qr3	= NULL;
	for (i = 0; i < num_edges; i++) {
		_gst_qr3_clear (&edge_qr3 [i]);
	}
	free (edge_qr3);

	fst_qr3		= gcp -> fst_qr3;
	gcp -> fst_qr3	= NULL;
	for (i = 0; i < num_fsts; i++) {
		_gst_qr3_clear (&fst_qr3 [i]);
	}
	free (fst_qr3);

	free (gcp -> edge_base);	gcp -> edge_base = NULL;
	free (gcp -> stein_base);	gcp -> stein_base = NULL;
}

/*
 * Virtual destructor for our "cost extension" object.
 */

	static
	void
_egmp_cext_free (

struct cost_extension *		extp
)
{
int				i, n;
struct _egmp_cost_extension *	xp;

	xp = (struct _egmp_cost_extension *) extp;

	FATAL_ERROR_IF (xp -> base.vtab NE &_egmp_cext_vtab);

	if (xp -> edge_costs NE NULL) {
		n = xp -> num_edges;
		for (i = 0; i < n; i++) {
			mpz_clear (&(xp -> edge_costs [i]));
		}
		free (xp -> edge_costs);
		xp -> edge_costs = NULL;
	}

	xp -> base.vtab = NULL;

	free (xp);
}

/*
 * Virtual function to write out dummy cost values.
 */

	static
	void
_egmp_cext_write_edge_cost (

struct cost_extension *		extp,
FILE *				fp,
int				i
)
{
int				n;
struct _egmp_cost_extension *	xp;

	FATAL_ERROR_IF ((extp EQ NULL) OR (fp EQ NULL));

	xp = (struct _egmp_cost_extension *) extp;

	FATAL_ERROR_IF (xp -> base.vtab NE &_egmp_cext_vtab);

	n = xp -> num_edges;
	FATAL_ERROR_IF ((i < 0) OR (i >= n));

	if (xp -> edge_costs EQ NULL) {
		fprintf (fp, "DUMMY_COST_FOR_EDGE(%d)", i);
	}
	else {
		mpz_out_str (fp, 10, &(xp -> edge_costs [i]));
	}
}

#endif

/*
 * Members of struct eqp_t that are used by the algorithms in this file:
 *
 *		E
 *		origin_term
 *		DV
 *		L
 *		R
 *
 * Members of struct eqp_t that are *NOT* used in this file:
 *
 *		index		(debugging code only)
 *		LP
 *		RP
 *		S		(debugging code only)
 *		UB
 *		BS
 *		DC
 *		DR
 *		DR2
 *		Z
 *		SMINX
 *		SMAXX
 *		SMINY
 *		SMAXY
 *		CHOSEN
 */
