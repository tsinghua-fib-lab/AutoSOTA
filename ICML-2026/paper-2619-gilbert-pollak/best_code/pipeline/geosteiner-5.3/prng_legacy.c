/***********************************************************************

	$Id: prng_legacy.c,v 1.5 2023/04/03 17:18:37 warme Exp $

	File:	prng_legacy.c
	Rev:	e-3
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A Pseudo-Random Number Generator for the "rand_points.c"
	program that is based on the (good old? bad old? just plain
	bad?) PDP-11 Unix rand() function.

	The badness of this generator is truly legendary.  But this
	generator is retained because some of my "favorite" test
	instances come from it.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Created.
	e-2:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.
	e-3:	04/03/2023	warme
		: Implement new gen_array interface.

************************************************************************/

#include "rand_points.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatal.h"
#include "logic.h"
#include "memory.h"


/*
 * The `opaque' data used by this PRNG implementation.
 */

struct my_data {
	/* A copy of the options we were given */
	struct PRNG_options	options;

	int32u			seed;
};

/*
 * Local Functions
 */

static int16u		__rand (struct my_data * mdp);
static struct PRNG_obj *
		create_instance (const struct PRNG_options * opts);
static void	free_instance (struct PRNG_obj * obj);
static void	gen_binary_array (struct my_data * mdp, int n, double * array);
static void	gen_binary_points (struct my_data * mdp, int n, FILE * fp);
static void	gen_decimal_array (struct my_data *	mdp,
				   int			npoints,
				   int			ndigits,
				   int			nplaces,
				   double *		array);
static void	gen_decimal_points (struct my_data *	mdp,
				    int			npoints,
				    int			ndigits,
				    int			nplaces,
				    FILE *		fp);
static int	generate_array (struct PRNG_obj * obj, double * array, int n);
static int	generate_points (struct PRNG_obj * obj, FILE * fp, int n);
static double	get_binary_double (struct my_data * mdp);
static int32u	random_coordinate (struct my_data * mdp, int max_coord);
static bool	read_state_from_stream (struct PRNG_obj * obj, FILE * fp);
static int	write_state_to_stream (struct PRNG_obj * obj, FILE * fp);

/*
 * Create and return an instance of this PRNG implementation having
 * the given options.  Returns NULL (after printing an error message)
 * if this implementation does not like any of the given options.
 */

	static
	struct PRNG_obj *
create_instance (

const struct PRNG_options *	opts	/* IN: Options to use */
)
{
int			ndigits, nplaces;
struct PRNG_obj *	p;
struct my_data *	mdp;

	FATAL_ERROR_IF (opts EQ NULL);

	ndigits = opts -> ndigits;
	nplaces = opts -> nplaces;

	/* Default is 4-digit integers. */
	if (ndigits < 0) {
		ndigits = 4;
		if (nplaces < 0) {
			nplaces = 0;
		}
	}

	switch (opts -> mode) {
	case MODE_DECIMAL:
		FATAL_ERROR_IF (ndigits < 1);

		if (ndigits > 5) {
			fprintf (stderr,
				 "The `legacy' generator supports"
				 " at most 5 digit decimal numbers\n");
			return (NULL);
		}
		break;

	case MODE_BINARY:
		break;

	default:
		FATAL_ERROR;
		return (NULL);
	}

	if (opts -> key NE NULL) {
		fprintf (stderr,
			 "The `legacy' generator (-g %d) does not support the"
			 " -k option.\n",
			 PRNG_IMP_LEGACY);
		return (NULL);
	}

	/* Create our private data block. */
	mdp = NEW (struct my_data);
	memset (mdp, 0, sizeof (*mdp));

	mdp -> options	= *opts;
	mdp -> options.ndigits	= ndigits;
	mdp -> options.nplaces	= nplaces;
	mdp -> options.key = NULL;	/* Might become invalid, so ignore. */

	/* Set default random seed. */
	mdp -> seed = 1;

	if (mdp -> options.randomize) {
		mdp -> seed = mdp -> options.cur_time;
	}

	p = NEW (struct PRNG_obj);
	p -> type		= PRNG_IMP_LEGACY;
	p -> opaque		= (void *) mdp;
	p -> read_seed		= read_state_from_stream;
	p -> gen_points		= generate_points;
	p -> gen_array		= generate_array;
	p -> write_seed		= write_state_to_stream;
	p -> destruct		= free_instance;

	return (p);
}

/*
 * Free up the given instance.
 */

	static
	void
free_instance (

struct PRNG_obj *	obj	/* IN: instance to free */
)
{
struct my_data *	mdp;

	if (obj EQ NULL) {
		return;
	}

	mdp = (struct my_data *) (obj -> opaque);

	free (mdp);

	free (obj);
}

/*
 * Generate the given number of points.
 */

	static
	int
generate_points (

struct PRNG_obj *	obj,		/* IN/OUT: our instance object */
FILE *			fp,		/* IN/OUT: output stream to write to */
int			n		/* IN: number of points to generate */
)
{
struct my_data *	mdp;

	FATAL_ERROR_IF ((obj EQ NULL) OR (fp EQ NULL));

	mdp = (struct my_data *) (obj -> opaque);

	switch (mdp -> options.mode) {
	case MODE_BINARY:
		gen_binary_points (mdp, n, fp);
		break;

	case MODE_DECIMAL:
		gen_decimal_points (mdp,
				    n,
				    mdp -> options.ndigits,
				    mdp -> options.nplaces,
				    fp);
		break;

	default:
		FATAL_ERROR;
		break;
	}

	if (ferror (fp)) {
		return (EIO);
	}

	return (0);
}

/*
 * Generate points in binary mode.
 */

	static
	void
gen_binary_points (

struct my_data *	mdp,	/* IN/OUT: my private data */
int			n,	/* IN: number of points to generate */
FILE *			fp	/* IN/OUT: file to write into */
)
{
int		i;
double		x, y;

	/* We get 128-bit words from our random stream, each of	*/
	/* which gives us 2 64-bit mantissas, or one point!	*/

	for (i = 0; i < n; i++) {
		x = get_binary_double (mdp);
		y = get_binary_double (mdp);

		fprintf (fp, "%.17g\t%.17g\n", x, y);
	}
}

/*
 * Generate a random IEEE double in the range [0,1).
 */

	static
	double
get_binary_double (

struct my_data *	mdp	/* IN/OUT: my private data */
)
{
int		i;
int64u		mant;
double		x;

	mant = 0;
	for (i = 0; i < 6; i++) {
		mant <<= 10;
		mant ^= __rand (mdp);
	}

	x = mant;
	x = ldexp (x, -64);

	return (x);
}

/*
 * Generate points in decimal mode.
 */

	static
	void
gen_decimal_points (

struct my_data *	mdp,	/* IN/OUT: my private data */
int			n,	/* IN: number of points to generate */
int			ndigits,/* IN: number of digits per coordinate */
int			nplaces,/* IN: number of those digits that are */
				/*     fractional (i.e., lie to the */
				/*     right of the decimal point) */
FILE *			fp	/* IN/OUT: file to write into */
)
{
int		i, part1, part2, max_part1, max_part2;
int32u		x, y;

	if (nplaces < 0) {
		/* This means all digits are fractional. */
		nplaces = ndigits;
	}
	part1 = ndigits - nplaces;
	part2 = nplaces;

	max_part1 = 1;
	for (i=0; i < part1; i++) {
		max_part1 *= 10;
	}

	max_part2 = 1;
	for (i=0; i < part2; i++) {
		max_part2 *= 10;
	}

	for (i = 0; i < n; i++) {
		x = random_coordinate (mdp, max_part1);
		fprintf (fp, "%*u", part1 + 1, x);
		if (part2 > 0) {
			__rand (mdp);		/* Discard another number. */
			x = random_coordinate (mdp, max_part2);
			fprintf (fp, ".%0*u", part2, x);
		}

		__rand (mdp);	/* Discard another number. */

		y = random_coordinate (mdp, max_part1);
		fprintf (fp, " %*u", part1 + 1, y);
		if (part2 > 0) {
			__rand (mdp);		/* Discard another number. */
			y = random_coordinate (mdp, max_part2);
			fprintf (fp, ".%0*u", part2, y);
		}

		fprintf (fp, "\n");
	}
}

/*
 * Generate random points into a given array.
 */

	static
	int
generate_array (

struct PRNG_obj *	obj,		/* IN/OUT: our instance object */
double *		array,		/* IN/OUT: Array to fill in */
int			n		/* IN: number of points to generate */
)
{
struct my_data *	mdp;

	FATAL_ERROR_IF ((obj EQ NULL) OR (array EQ NULL));

	mdp = (struct my_data *) (obj -> opaque);

	switch (mdp -> options.mode) {
	case MODE_BINARY:
		gen_binary_array (mdp, n, array);
		break;

	case MODE_DECIMAL:
		gen_decimal_array (mdp,
				   n,
				   mdp -> options.ndigits,
				   mdp -> options.nplaces,
				   array);
		break;

	default:
		FATAL_ERROR;
		break;
	}

	return (0);
}

/*
 * Generate array in binary mode.
 */

	static
	void
gen_binary_array (

struct my_data *	mdp,	/* IN/OUT: my private data */
int			n,	/* IN: number of points to generate */
double *		array	/* IN/OUT: Array to fill in */
)
{
int		i;
double		x, y;

	/* We get 128-bit words from our random stream, each of	*/
	/* which gives us 2 64-bit mantissas, or one point!	*/

	for (i = 0; i < n; i++) {
		x = get_binary_double (mdp);
		y = get_binary_double (mdp);

		*array++	= x;
		*array++	= y;
	}
}

/*
 * Generate points in decimal mode.
 */

	static
	void
gen_decimal_array (

struct my_data *	mdp,	/* IN/OUT: my private data */
int			n,	/* IN: number of points to generate */
int			ndigits,/* IN: number of digits per coordinate */
int			nplaces,/* IN: number of those digits that are */
				/*     fractional (i.e., lie to the */
				/*     right of the decimal point) */
double *		array	/* IN/OUT: Array to fill in */
)
{
int		i, part1, part2, max_part1, max_part2;
int32u		x, y, scale;

	if (nplaces < 0) {
		/* This means all digits are fractional. */
		nplaces = ndigits;
	}

	/* NOTE: If nplaces > 0, we get rounding errors! */

	part1 = ndigits - nplaces;
	part2 = nplaces;

	scale = 1.0;
	for (i = 0; i < part2; i++) {
		scale *= 10.0;
	}

	max_part1 = 1;
	for (i=0; i < part1; i++) {
		max_part1 *= 10;
	}

	max_part2 = 1;
	for (i=0; i < part2; i++) {
		max_part2 *= 10;
	}

	for (i = 0; i < n; i++) {
		x = random_coordinate (mdp, max_part1);
		if (part2 > 0) {
			__rand (mdp);		/* Discard another number. */
			x = x * scale + random_coordinate (mdp, max_part2);
			x /= scale;	/* Rounding! */
		}
		*array++ = x;

		__rand (mdp);	/* Discard another number. */

		y = random_coordinate (mdp, max_part1);
		if (part2 > 0) {
			__rand (mdp);		/* Discard another number. */
			y = y * scale + random_coordinate (mdp, max_part2);
			y /= scale;	/* Rounding! */
		}
		*array++ = y;
	}
}

/*
 * This routine computes a new random coordinate.  This is done in one
 * of two ways:  using the "bad" PDP-11 Unix method, or using a better
 * 64-bit shift register with XOR feedback.  (For backward compatibility,
 * the old way is the default...)
 */

	static
	int32u
random_coordinate (

struct my_data *	mdp,		/* IN/OUT: my private data */
int			max_coord	/* IN: modulus for coordinates */
)
{
int32u		n, r1, r2;

	/* Doing it the "bad" old way... */

	r1 = __rand (mdp);
	r2 = __rand (mdp);

	n = (r1 >> 2) ^ (r2 << 3);

	return (n % max_coord);
}

/*
 * This routine produces random numbers the "bad" way (i.e. the way
 * the original PDP-11 Unix did).
 */

	static
	int16u
__rand (

struct my_data *	mdp	/* IN/OUT: my private data */
)
{
long		x;

	x = 1103515245L * ((long) mdp -> seed) + 12345L;
	mdp -> seed = x;

	return ((x >> 16) & 0x7FFF);
}

/*
 * Write our PRNG state out to the given output stream.
 */

	static
	int
write_state_to_stream (

struct PRNG_obj *	obj,		/* IN: Our instance object */
FILE *			fp		/* IN/OUT: output stream to write to */
)
{
struct my_data *	mdp;

	FATAL_ERROR_IF (obj EQ NULL);

	mdp = (struct my_data *) (obj -> opaque);

	fprintf (fp, "%08x\n", mdp -> seed);

	if (ferror (fp)) {
		return (EIO);
	}

	return (0);
}

/*
 * Return true if-and-only-if a valid PRNG state can be successfully read
 * from the given input stream into our PRNG state object.
 */

	static
	bool
read_state_from_stream (

struct PRNG_obj *	obj,		/* IN/OUT: our instance object */
FILE *			fp		/* IN: input stream to read from */
)
{
int			i;
int32u			seed;
char			c;
struct my_data *	mdp;

	FATAL_ERROR_IF ((obj EQ NULL) OR (fp EQ NULL));

	mdp = (struct my_data *) (obj -> opaque);

	i = fscanf (fp, " %x", &seed);
	if (i NE 1) return (FALSE);

	mdp -> seed = seed;

	c = fgetc (fp);
	if (c NE EOF) return (FALSE);

	/* The file appears to be consistent. */

	return (TRUE);
}

/*
 * The public implementation object exposed by this implementation.
 */

const struct PRNG_imp	rand_points_PRNG_legacy = {
	.available	= TRUE,
	.name		= "legacy",
	.create		= create_instance,
};
