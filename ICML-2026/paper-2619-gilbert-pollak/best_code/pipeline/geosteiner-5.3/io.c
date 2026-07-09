/***********************************************************************

	$Id: io.c,v 1.31 2023/04/03 17:27:58 warme Exp $

	File:	io.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for Input/Output of point sets and coordinates.
	These routines insulate the rest of the system from the
	exact data type for "coord_t".

	To eliminate errors due to the basic properties of binary
	floating point representation, we scale (i.e., multiply)
	every coordinate by a power of 10 until every coordinate has
	an integral value.  This is done PRIOR to conversion into
	floating point form, and permits us to represent each
	coordinate (as well as delta X and delta Y values) exactly.
	This actually eliminates all numeric errors when computing
	and summing distances in the rectilinear metric (as long as
	the numbers do not exceed the basic precision of a double).

************************************************************************

	Modification Log:

	a-1:	02/20/93	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Moved Data_Num_Decimal_Places and min_precision
		:  into new struct scale_info.  Now passing this
		:  scale_info explicitly everywhere it is needed.
		: Changed "struct numlist" to be a partially converted
		:  scaled numeric form rather than textual form.
		: Accept scientific notation and leading + signs.
		: Support negative scale factors.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Added some functions for scaling.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix -Wall issue.
	e-5:	04/03/2023	warme
		: Set status (if requested).

************************************************************************/

#include "io.h"

#include <ctype.h>
#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>

/*
 * Global Routines
 */

int			gst_get_points (FILE *, int, double **, gst_scale_info_ptr);

int			_gst_compute_scaling_factor (struct numlist * list);
void			_gst_coord_to_string (char *			buf,
					      coord_t			coord,
					      gst_scale_info_ptr	sip);
void			_gst_dist_to_string (char *			buf,
					     dist_t			dist,
					     gst_scale_info_ptr		sip);
int			gst_compute_scale_info_digits (int,
						double *,
						gst_scale_info_ptr);
char *			gst_unscale_to_string (char *,
					       double,
					       gst_scale_info_ptr);
int			gst_get_hg_scale_info (gst_hg_ptr,
					       gst_scale_info_ptr *);
int			gst_set_hg_scale_info (gst_hg_ptr,
					       gst_scale_info_ptr);
struct numlist *	_gst_parse_line_of_numbers (FILE * fp);
coord_t			_gst_read_numlist (struct numlist *	nlp,
					   gst_scale_info_ptr	sip);
void			_gst_set_scale_info (gst_scale_info_ptr	sip,
					     int		scale_factor);
double			gst_unscale_to_double (double, gst_scale_info_ptr);


/*
 * Local Routines
 */

static struct numlist *	new_numlist (double, int, int);
static int		next_char (FILE *);
static struct numlist *	parse_input_numbers (FILE *, int);
static struct numlist *	parse_one_number (FILE *, bool);


/*
 * Local Variables
 */

static const struct gst_scale_info default_scale_info = { 0, DBL_DIG + 1, 1, 1};

/*
 * This routine reads in a set of points from the standard input.
 *
 * It only allocates memory itself if maxpoints is 0.
 */

	int
gst_get_points (

FILE *			fp,		/* IN - stream to read from */
int			maxpoints,	/* IN - maximum number of points (0 = infty) */
double			**points,	/* OUT - list of coordinates */
gst_scale_info_ptr	sip		/* OUT - problem scaling info */
)
{
int			n;
int			scaling_factor;
double *		p;
coord_t			x;
coord_t			y;
struct numlist *	list;
struct numlist *	nlp;

	GST_PRELUDE

	/* Gather all numbers from the input in an accurate scaled form. */

	list = parse_input_numbers (fp, 2*maxpoints);

	if (sip NE NULL) {
		/* Determine the coordinate scaling factor to use. */
		scaling_factor = _gst_compute_scaling_factor (list);
		_gst_set_scale_info (sip, scaling_factor);
	}

	n = 0;
	for (nlp = list; nlp NE NULL; nlp = nlp -> next) {
		++n;
	}

	if ((n & 1) NE 0) {
		(void) fprintf (stderr, "X coord without Y coord.\n");
		exit (1);
	}
	n >>= 1;

	if (points NE NULL) {
		if (maxpoints EQ 0) {
			*points = NEWA (2*n, double);

			/* Zero out the entire point set. */
			memset (*points, 0, 2*n*sizeof(double));
		}

		p = *points;

		while (list NE NULL) {
			nlp = list;
			list = list -> next;

			x = _gst_read_numlist (nlp, sip);

			FATAL_ERROR_IF (list EQ NULL);

			nlp = list;
			list = list -> next;

			y = _gst_read_numlist (nlp, sip);

			*(p++)		= x;
			*(p++)		= y;
		}
	}

	GST_POSTLUDE
	return (n);
}

/*
 * This routine "parses" all of the numbers in the input into a list
 * of partially converted/scaled numbers in binary (floating point) form.
 */

	static
	struct numlist *
parse_input_numbers (

FILE *		fp,		/* IN - input stream to read from */
int		maxnum		/* IN - maximum number of numbers (0 = infty) */
)
{
struct numlist *	p;
struct numlist *	root;
struct numlist **	hookp;
int			num;

	root = NULL;
	hookp = &root;
	num = 0;

	for (;;) {
		p = parse_one_number (fp, FALSE);
		if (p EQ NULL) break;

		*hookp = p;
		hookp = &(p -> next);
		num++;
		if (num == maxnum) break;
	}

	return (root);
}

/*
 * Read a single text line of numbers.  We ignore blank lines and
 * lines containing only a comment.  Otherwise, this routine collects
 * numbers in a linked list of numlist structures until the first
 * newline after at least one number is read.
 */

	struct numlist *
_gst_parse_line_of_numbers (

FILE *		fp		/* IN - file to read from */
)
{
struct numlist *	root;
struct numlist **	hookp;
struct numlist *	p;

	root = NULL;
	hookp = &root;

	for (;;) {
		p = parse_one_number (fp, TRUE);
		if (p NE NULL) {
			*hookp = p;
			hookp = &(p -> next);
			continue;
		}

		/* We read either EOF or a newline... */
		if (root NE NULL) {
			/* we have one or more numbers -- we don't	*/
			/* really care whether it was newline or EOF!	*/
			break;
		}
		if (feof (fp)) {
			/* It was EOF.  We are not going to get any	*/
			/* more numbers!				*/
			break;
		}
		/* ignore blank/comment line. */
	}

	return (root);
}

/*
 * This routine "parses" the next number from the input stream, and converts
 * it into binary form.  This form is "integerized" to minimize numeric
 * error caused by fractional digits.
 */

	static
	struct numlist *
parse_one_number (

FILE *		fp,		/* IN - input stream to read from */
bool		find_newline	/* IN - fail if newline seen before number */
)
{
int		num_dp;
int		nsig;
int		nzero;
int		expon;
int		esign;
bool		dot_flag;
int		c;
double		sign;
double		num;
double		fnum;
double		fpow;

#define MAX_INT_DIGITS	14

	if (feof (fp)) return (NULL);

	sign = 1.0;

	do {
		c = next_char (fp);
		if (c < 0) return (NULL);
		if (c EQ '.') break;
		if (c EQ '-') {
			/* Note the minus sign and proceed. */
			sign = -1.0;
			c = next_char (fp);
			break;
		}
		if (c EQ '+') {
			c = next_char (fp);
			break;
		}
		if (ispunct (c)) {
			/* Strip comment line... */
			do {
				c = next_char (fp);
				if (c < 0) return (NULL);
			} while (c NE '\n');
			if (find_newline) return (NULL);
			continue;
		}
		if ((c EQ '\n') AND find_newline) return (NULL);
	} while (NOT isdigit (c));

	num = 0.0;
	fnum = 0.0;
	fpow = 1.0;

	nsig = 0;
	nzero = 0;
	num_dp = 0;
	dot_flag = FALSE;

	for (;;) {
		if (c EQ '.') {
			if (dot_flag) {
				/* OOPS!  More than 1 decimal point	*/
				/* seen in a single number...  Just	*/
				/* terminate this number.		*/
				ungetc (c, fp);
				break;
			}
			dot_flag = TRUE;
		}
		else if (c EQ '0') {
			if (dot_flag) {
				++num_dp;
			}
			if (nsig <= 0) {
				/* Ignore: no non-zero digits yet seen. */
			}
			else {
				/* Don't process these if we don't have to. */
				++nzero;
			}
		}
		else if (isdigit (c)) {
			/* Process any deferred zero digits... */
			while (nzero > 0) {
				if (nsig < MAX_INT_DIGITS) {
					num *= 10.0;
					++nsig;
				}
				else {
					fpow *= 10.0;
					if (dot_flag) {
						--num_dp;
					}
				}
				--nzero;
			}
			if (nsig < MAX_INT_DIGITS) {
				num = 10.0 * num + (c - '0');
				++nsig;
				if (dot_flag) {
					++num_dp;
				}
			}
			else {
				fpow *= 10.0;
				fnum += (c - '0') / fpow;
			}
		}

		c = next_char (fp);
		if (c < 0) break;

		if ((NOT isdigit (c)) AND (c NE '.')) break;
	}

	expon = 0;

	if ((c EQ 'e') OR (c EQ 'E')) {
		/* Read the exponent. */
		esign = 1;
		c = next_char (fp);
		if (c EQ '+') {
			c = next_char (fp);
		}
		else if (c EQ '-') {
			esign = -1;
			c = next_char (fp);
		}
		while ((c >= 0) AND isdigit (c)) {
			expon = 10 * expon + (c - '0');
			c = next_char (fp);
		}
		expon *= esign;
	}
	if (c >= 0) {
		ungetc (c, fp);
	}

	/* Put sign, integral, and fractional parts together. */
	num = sign * (num + fnum);

	expon += nzero;
	expon -= num_dp;

	return (new_numlist (num, expon, nsig));
}

/*
 * This routine creates a scaled number list structure.
 */

	static
	struct numlist *
new_numlist (

double		mantissa,	/* IN - scaled mantissa value */
int		expon,		/* IN - exponent (power of 10) */
int		nsig		/* IN - num significant mantissa digits */
)
{
struct numlist *	p;

	p = NEW (struct numlist);

	p -> mantissa	= mantissa;
	p -> expon	= expon;
	p -> nsig	= (nsig > 0) ? nsig : 1;
	p -> next	= NULL;

	return (p);
}

/*
 * Compute a common scale value for all of the numbers in the given list.
 * In most cases, we can obtain an "optimum" scale value -- wherein every
 * number in the list will have an integer value that can be represented
 * exactly.  In cases where there are many significant digits, or a wide
 * span of exponent values, we must compromise.
 */

	int
_gst_compute_scaling_factor (

struct numlist *	p	/* IN - list of numbers to scale */
)
{
int		k;
int		min_expon;
int		max_expon;

	for (;;) {
		if (p EQ NULL) {
			/* No non-zero numbers -- don't scale at all! */
			return (0);
		}
		if (p -> mantissa NE 0.0) break;
		p = p -> next;
	}

	min_expon = p -> expon;
	max_expon = p -> expon + p -> nsig;

	for (p = p -> next; p NE NULL; p = p -> next) {
		if (p -> mantissa EQ 0) continue;
		if (p -> expon < min_expon) {
			min_expon = p -> expon;
		}
		k = p -> expon + p -> nsig;
		if (k > max_expon) {
			max_expon = k;
		}
	}

	/* Let D be a significant digit from the list, E be its		*/
	/* exponent so that the digit has value D * 10**E.  We now know	*/
	/* the smallest and largest of the E values.  If the spread is	*/
	/* MAX_INT_DIGITS or fewer, we win!  (Use the smallest E.)	*/
	/* Otherwise, try to split the difference.			*/

	max_expon -= MAX_INT_DIGITS;

	if (max_expon > min_expon) {
		/* Must compromise. */
		min_expon = (max_expon + min_expon + 1) / 2;
	}

	/* Our internal representation is such that dividing it by	*/
	/* 10**-min_expon yields external values.  (Note that min_expon	*/
	/* is typically negative here, in which case we would really be	*/
	/* multiplying by 10**(-min_expon).)				*/

	return (-min_expon);
}

	gst_scale_info_ptr
gst_create_scale_info (

int		*status		/* OUT - */
)
{
gst_scale_info_ptr	sip;

	GST_PRELUDE

	sip = NEW (struct gst_scale_info);
	memcpy (sip, &default_scale_info, sizeof(default_scale_info));

	if (status NE NULL) {
		*status = 0;
	}

	GST_POSTLUDE
	return (sip);
}

	int
gst_free_scale_info (

gst_scale_info_ptr	scinfo	/* IN - structure to be freed */
)
{

	GST_PRELUDE

	free (scinfo);
	GST_POSTLUDE
	return (0);
}

/*
 * Set the scaling info properly for the given scale factor.
 */

	void
_gst_set_scale_info (

gst_scale_info_ptr	sip,		/* OUT - scale_info to initialize */
int			scale_factor	/* IN - scale factor to establish */
)
{
int		i;
double		pow10;
double		p;

	memset (sip, 0, sizeof (*sip));

	sip -> min_precision	= DBL_DIG + 1;
	sip -> scale		= scale_factor;

	i = scale_factor;
	if (i < 0) {
		i = - i;
	}

	/* Compute p = 10 ** i. */
	pow10 = 1.0;
	p = 10.0;
	while (i > 0) {
		if ((i & 1) NE 0) {
			pow10 *= p;
		}
		p *= p;
		i >>= 1;
	}

	if (scale_factor < 0) {
		sip -> scale_mul = pow10;
		sip -> scale_div = 1.0;
	}
	else {
		sip -> scale_mul = 1.0;
		sip -> scale_div = pow10;
	}
}

/*
 * This routine converts the coordinate in the given numlist structure
 * to an internal coordinate value, scaled as specified by the scaling_factor
 * parameter.  The given numlist structure is freed.
 */

	coord_t
_gst_read_numlist (

struct numlist *	nlp,		/* IN - string list structure */
gst_scale_info_ptr	sip		/* IN - problem scaling info */
)
{
int		i;
int		target_expon;
int		expon;
coord_t		value;
double		pow10;
double		p;

/******** IS THIS THE CORRECT HANDLING OF A NULL POINTER ********/
	target_expon = sip ? -(sip -> scale) : 0;

	value = nlp -> mantissa;
	expon = nlp -> expon;

	i = expon - target_expon;
	if (i < 0) {
		i = - i;
	}
	pow10 = 1.0;
	p = 10.0;
	while (i > 0) {
		if ((i & 1) NE 0) {
			pow10 *= p;
		}
		p *= p;
		i >>= 1;
	}
	if (expon > target_expon) {
		value *= pow10;
	}
	else if (expon < target_expon) {
		value /= pow10;
	}

	free ((char *) nlp);

	return (value);
}

/*
 * This routine gets the next character from stdin.  For the case of
 * interactive input, it makes sure that an EOF condition is
 * "permanent" -- that is, getc will not be called again after it
 * indicates EOF.
 */

	static
	int
next_char (

FILE *		fp		/* IN - input stream to read from */
)
{
int		c;

	if (feof (fp)) {
		/* Don't do "getc" again -- EOF is permanent! */
		return (EOF);
	}

	c = getc (fp);

	return (c);
}

/*
 * This routine converts the given internal-form coordinate to a
 * printable ASCII string in external form.  The internal form is an
 * integer (to eliminate numeric problems), but the external data
 * may actually involve decimal fractions.  This routine therefore
 * properly scales the coordinate during conversion to printable form.
 */

	char *
gst_unscale_to_string (

char *			buf,	/* OUT - buffer to put ASCII text into */
double			val,	/* IN - double to convert */
gst_scale_info_ptr	sip	/* IN - problem scaling info */
)
{
	GST_PRELUDE

	if (sip) {
		_gst_dist_to_string (buf, val, sip);
	}
	else {
		sprintf (buf, "%.15f", val);
	}

	GST_POSTLUDE
	return buf;
}

/*
 * This routine converts the given internal-form coordinate to a
 * printable ASCII string in external form.  The internal form is an
 * integer (to eliminate numeric problems), but the external data
 * may actually involve decimal fractions.  This routine therefore
 * properly scales the coordinate during conversion to printable form.
 */

	void
_gst_coord_to_string (

char *			buf,	/* OUT - buffer to put ASCII text into */
coord_t			coord,	/* IN - coordinate to convert */
gst_scale_info_ptr	sip	/* IN - problem scaling info */
)
{
	/* Just pretend its a distance -- distances certainly	*/
	/* do not have LESS precision than coordinates...	*/
	_gst_dist_to_string (buf, coord, sip);
}

/*
 * This routine converts the given internal-form distance to a
 * printable ASCII string in external form.  The internal form is an
 * integer (to eliminate numeric problems), but the external data
 * may actually involve decimal fractions.  This routine therefore
 * properly scales the distance during conversion to printable form.
 */

	void
_gst_dist_to_string (

char *			buf,	/* OUT - buffer to put ASCII text into */
dist_t			dist,	/* IN - distance to convert */
gst_scale_info_ptr	sip	/* IN - problem scaling info */
)
{
int		i;
int		digit;
int		mindig;
double		ipart;
double		fpart;
double		div10;
char *		p;
char *		endp;
char		tmp [256];

	if (dist < 0.0) {
		dist = -dist;
		*buf++ = '-';
	}
	if (dist EQ 0) {
		*buf++ = '0';
		*buf++ = '\0';
		return;
	}

	mindig = sip -> min_precision;

	memset (tmp, '0', sizeof (tmp));
	p = &tmp [128];
	endp = p;

	ipart = floor (dist);
	while (ipart > 0) {
		div10 = floor (ipart / 10.0);
		digit = ipart - floor (10.0 * div10 + 0.5);
		*--p = digit + '0';
		ipart = div10;
	}

	if ((sip -> scale <= 0) AND (mindig EQ 0)) {
		/* The coordinates and edge costs are all integers	*/
		/* (both internally and externally so scaling is a	*/
		/* "no-op").  The metric is not Euclidean, so nothing	*/
		/* irrational appears.  Always output integers, and do	*/
		/* NOT insert a decimal point...			*/

		endp [-(sip -> scale)] = '\0';
		strcpy (buf, p);
		return;
	}

	if (sip -> scale >= 0) {
		if (p + sip -> scale > endp) {
			p = endp - sip -> scale;
		}

		for (i = 0; i <= sip -> scale; i++) {
			endp [1 - i] = endp [0 - i];
		}
		endp [1 - i] = '.';
		++endp;

		fpart = dist - floor (dist);
		while ((endp - p) < (mindig + 1)) {
			fpart *= 10.0;
			digit = (int) floor (fpart);
			*endp++ = digit + '0';
			fpart -= ((double) digit);
		}
	}
	else if (mindig EQ 0) {
		endp -= sip -> scale;
	}
	else {
		fpart = dist - floor (dist);
		for (i = sip -> scale; i < 0; i++) {
			fpart *= 10.0;
			digit = (int) floor (fpart);
			*endp++ = digit + '0';
			fpart -= ((double) digit);
			--mindig;
		}
		if (mindig > 0) {
			*endp++ = '.';
			while (mindig > 0) {
				fpart *= 10.0;
				digit = (int) floor (fpart);
				*endp++ = digit + '0';
				fpart -= ((double) digit);
				--mindig;
			}
		}
	}

	*endp = '\0';

	strcpy (buf, p);
}


/*
 * This routine sets up the various parameters needed for outputting
 * scaled coordinates.  We want to print coordinates/distances with
 * the minimum fixed precision whenever this gives the exact result.
 * Otherwise we will print the coordinates/distances with full
 * precision.
 *
 * The minimum fixed precision is exact ONLY when ALL of the following
 * are true:
 *
 *	- The metric must not be EUCLIDEAN, since distances become
 *	  irrational even if the coordinates are finite precision.
 *	  Coordinates of Euclidean Steiner points are also irrational.
 *
 *	- The SCALED vertex coordinates must all be integral.
 *
 *	- The SCALED hyperedge costs must all be integral.
 *
 * Note: we actually check the scaled data for integrality because there
 * are some old FST generators that do not implement scaling!  Such
 * data always contain a scale factor of zero and non-integral coordinates
 * and distances.
 */

	int
gst_compute_scale_info_digits (

int			npts,		/* IN - number of terminals */
double *		pts,		/* IN - list of terminals */
gst_scale_info_ptr	sip		/* IN/OUT - problem scaling info */
)
{
int			i;
double *		p;
bool			integral;
double			c;

	GST_PRELUDE

	integral = TRUE;

	if (pts NE NULL) {
		p = pts;
		for (i = 0; i < npts; i++) {
			double x = *p++;
			double y = *p++;

			c = floor (x);
			if (c NE x) {
				integral = FALSE;
				break;
			}
			c = floor (y);
			if (c NE y) {
				integral = FALSE;
				break;
			}
		}
	}

	/* If integral */
	if (integral) {
		sip -> min_precision = 0;
	}
	else {
		sip -> min_precision = DBL_DIG + 1;
	}

	GST_POSTLUDE
	return (0);
}

	int
gst_set_hg_scale_info (

gst_hg_ptr		H,
gst_scale_info_ptr	scinfo
)
{
	GST_PRELUDE

	if (scinfo EQ NULL) {
		scinfo = (gst_scale_info_ptr)&default_scale_info;
	}

	memcpy (H -> scale, scinfo, sizeof (struct gst_scale_info));

	GST_POSTLUDE
	return (0);
}

	int
gst_get_hg_scale_info (

gst_hg_ptr		H,
gst_scale_info_ptr *	scinfo
)
{
	GST_PRELUDE

	*scinfo = H -> scale;

	GST_POSTLUDE
	return (0);
}

	double
gst_unscale_to_double (

double			val,
gst_scale_info_ptr	sip
)
{
double unscaled;

	GST_PRELUDE

	if (sip NE NULL) {
		unscaled = UNSCALE (val, sip);
	}
	else {
		unscaled = val;
	}

	GST_POSTLUDE
	return (unscaled);
}
