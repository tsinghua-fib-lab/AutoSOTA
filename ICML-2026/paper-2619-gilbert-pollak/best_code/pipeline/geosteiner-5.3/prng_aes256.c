/***********************************************************************

	$Id: prng_aes256.c,v 1.6 2023/04/03 17:19:50 warme Exp $

	File:	prng_aes256.c
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A Pseudo-Random Number Generator for the "rand_points.c"
	program that uses AES-256 as its principal source of
	randomness.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Created.
	e-2:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.
	e-3:	04/09/2018	warme
		: Added missing include file.
	e-4:	04/03/2023	warme
		: Implement new gen_array interface.

************************************************************************/

#include "rand_points.h"

#include "config.h"
#include "logic.h"

#if defined(HAVE_GMP)
#include <gmp.h>

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fatal.h"
#include "gstaes256.h"
#include "memory.h"

/*
 * An object containing all of the state for the core AES-256
 * Pseudo-Random Number Generator.
 *
 * It works by using AES-256 in CBC mode (Cipher Block Chaining).
 * The plain text is driven by a simple Linear Congruential Generator.
 */

struct PRNG {
	/* Linear congruential generator:				*/
	/*	LC(N) = (A + B*N) mod 2**128				*/
	/*								*/
	/* where A and B are constants, and n = 0, 1, 2, ...		*/
	mpz_t	a;
	mpz_t	b;
	mpz_t	n;
	mpz_t	lc;

	/* The AES-256 ECB state. */
	struct _gst_aes256_context	ctx;

	/* The Initialization Vector (IV) used to perform CBC --	*/
	/* Cipher Block Chaining.					*/
	int8u	iv [16];
};

/*
 * A state object for generating random integers having K decimal digits.
 *
 * Note that 2**128 is a 39 digit integer, so we can reasonably extract
 * at most 38 digits per block.
 */

#define	MAX_DIGITS_PER_BLOCK	38

struct DecBuf {
	struct PRNG *	state;		/* PRNG state object */
	int		ndigits;	/* # digits in generated numbers */
	int		K;		/* # of numbers extracted from */
					/* each block of 128 random bits */
	int		N;		/* Total number of decimal digits */
					/* extracted per random block */
	int8u		limit [16];	/* Re-draw random block if value */
					/* is >= this limit */
	int		nbuf;		/* How many numbers in buf[] */
	int64u		buf [MAX_DIGITS_PER_BLOCK];
					/* Set of numbers extracted from */
					/* one random block */
};

/*
 * The `opaque' data used by this PRNG implementation.
 */

struct my_data {
	/* A copy of the options we were given */
	struct PRNG_options	options;

	/* The core PRNG state object. */
	struct PRNG		prng;

	/* A special buffer (used only in `decimal' mode) for carving	*/
	/* 128-bit data blocks into sequences of uniformly distributed	*/
	/* N-digit decimal numbers.					*/
	struct DecBuf		dbuf;
};

/*
 * Local Functions
 */

static void	blank_out_leading_zeros (char * buf, int n);
static void	cleanup_state (struct PRNG * state);
static void	compute_next_random_block (int8u * buf, struct PRNG * state);
static void	convert_to_decimal (char *	buf,
				    int64u	value,
				    int		ndigits,
				    int		nplaces);
static struct PRNG_obj *
		create_instance (const struct PRNG_options * opts);
static void	free_instance (struct PRNG_obj * obj);
static void	gen_binary_array (struct PRNG * state, int n, double * array);
static void	gen_binary_points (struct PRNG * state, int n, FILE * fp);
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
static double	get_binary_double (const int8u * octets);
static int64u	get_decimal_number (struct DecBuf * dp);
static int	hexdig (char c);
static void	init_dbuf (struct DecBuf *	dp,
			   struct PRNG *	state,
			   int			ndigits);
static void	init_state (struct my_data * mdp, const char * key);
static void	my_mpz_set_hex (mpz_ptr dest, const char * hexval);
static int64u	put_dec_digs (char * buf, int64u value, int ndigits);
static bool	read_hex_octets (FILE * fp, int8u * octets, int n);
static bool	read_state_from_stream (struct PRNG_obj * obj, FILE * fp);
static void	refill_decimal_buffer (struct DecBuf * dp);
static void	write_hex_octets (FILE * fp, const int8u * octets, int n);
static int	write_state_to_stream (struct PRNG_obj * obj, FILE * fp);

/*
 * The default AES-256 key -- 256 bits from PI (the quintessential
 * "nothing up my sleeve" number).
 */

static const int8u	default_key [] = {
	0xC9, 0x0F, 0xDA, 0xA2, 0x21, 0x68, 0xC2, 0x34,
	0xC4, 0xC6, 0x62, 0x8B, 0x80, 0xDC, 0x1C, 0xD1,
	0x29, 0x02, 0x4E, 0x08, 0x8A, 0x67, 0xCC, 0x74,
	0x02, 0x0B, 0xBE, 0xA6, 0x3B, 0x13, 0x9B, 0x22
};

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
int			ndigits, maxdigs;
struct PRNG_obj *	p;
struct my_data *	mdp;

	FATAL_ERROR_IF (opts EQ NULL);

	/* This limit is imposed by int64u: 10**19 <= 2**64-1 < 10**20	*/
	maxdigs = 19;

	ndigits = opts -> ndigits;
	if (ndigits < 0) {
		ndigits = 7;
	}

	switch (opts -> mode) {
	case MODE_DECIMAL:
		FATAL_ERROR_IF (ndigits < 1);

		if (ndigits > maxdigs) {
			fprintf (stderr,
				 "The `AES-256' generator supports"
				 " at most %d digit decimal numbers\n",
				 maxdigs);
			return (NULL);
		}
		break;

	case MODE_BINARY:
		break;

	default:
		FATAL_ERROR;
		return (NULL);
	}

	/* Create our private data block. */
	mdp = NEW (struct my_data);
	memset (mdp, 0, sizeof (*mdp));

	mdp -> options	= *opts;
	mdp -> options.ndigits = ndigits;
	mdp -> options.key = NULL;	/* Might become invalid, so ignore. */

	/* Initialize the core Pseudo-Random Number Generator. */
	init_state (mdp, opts -> key);

	if (mdp -> options.mode EQ MODE_DECIMAL) {
		/* Initialize the Decimal Buffer too. */
		init_dbuf (&(mdp -> dbuf), &(mdp -> prng), ndigits);
	}

	p = NEW (struct PRNG_obj);
	p -> type		= PRNG_IMP_AES_256;
	p -> opaque		= (void *) mdp;
	p -> read_seed		= read_state_from_stream;
	p -> gen_points		= generate_points;
	p -> gen_array		= generate_array;
	p -> write_seed		= write_state_to_stream;
	p -> destruct		= free_instance;

	return (p);
}

/*
 * Initialize the Pseudo-Random Number Generator state.
 */

	static
	void
init_state (

struct my_data *	mdp,	/* IN/OUT: private data object to init */
const char *		key	/* IN: optional key (or NULL) */
)
{
int		i;
time_t		t;
size_t		size;
int		keylen;
struct PRNG *	state;
int8u		keybuf [32];

	FATAL_ERROR_IF (mdp EQ NULL);

	state = &(mdp -> prng);

	memset (state, 0, sizeof (*state));

	/* Init Linear Congruential Generator to default initial state. */
	mpz_init (state -> a);
	mpz_init (state -> b);
	mpz_init (state -> n);
	mpz_init (state -> lc);
	my_mpz_set_hex (state -> a, "7BCF299DD9BD46B09C370A85E3A50081");
	my_mpz_set_hex (state -> b, "8ACC8ADEBE4B518AE26409A1941855AB");
	mpz_set (state -> lc, state -> a);

	if (key EQ NULL) {
		/* User did not specify a key.  Use default. */
		memcpy (keybuf, default_key, 32);
	}
	else {
		size = strlen (key);
		keylen = (size > 32) ? 32 : ((int) size);
		memset (keybuf, 0, 32);
		memcpy (keybuf, key, keylen);
	}
	if (mdp -> options.randomize) {
		/* Fudge the key with given time value. */
		t = mdp -> options.cur_time;
		for (i = 31; i >= 0; i--) {
			keybuf [i] ^= (t & 0xFF);
			t >>= 8;
			if (t EQ 0) break;
		}
	}

	_gst_aes256_init (&(state -> ctx), keybuf);

	/* IV starts at zero. */
	memset (state -> iv, 0, 16);
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

	cleanup_state (&(mdp -> prng));

	/* The decimal buffer needs no cleanup. */

	free (mdp);

	free (obj);
}

/*
 * Clean up the final Pseudo-Random Number Generator state.
 */

	static
	void
cleanup_state (

struct PRNG *	state
)
{
	_gst_aes256_clear (&(state -> ctx));
	mpz_clear (state -> b);
	mpz_clear (state -> a);
	mpz_clear (state -> n);
	mpz_clear (state -> lc);

	memset (state, 0, sizeof (*state));
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
		gen_binary_points (&(mdp -> prng), n, fp);
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

struct PRNG *	state,		/* IN/OUT: random source */
int		n,		/* IN: number of points to generate */
FILE *		fp		/* IN/OUT: File to write into */
)
{
int		i;
int8u		buf [16];
double		x, y;

	/* We get 128-bit words from our random stream, each of	*/
	/* which gives us 2 64-bit mantissas, or one point!	*/

	for (i = 0; i < n; i++) {
		compute_next_random_block (buf, state);

		x = get_binary_double (buf);
		y = get_binary_double (buf + 8);

		fprintf (fp, "%.17g\t%.17g\n", x, y);
	}
}

/*
 * Convert the given block of 8 random octets (64-bits) into a
 * random IEEE double in the range [0,1).
 */

	static
	double
get_binary_double (

const int8u *	octets		/* IN: random octets to convert */
)
{
int32u		w1, w0;
double		x;

	/* Convert the octets into two 32-bit words. */
	w1 = 0;
	w1 |= *octets++;	w1 <<= 8;
	w1 |= *octets++;	w1 <<= 8;
	w1 |= *octets++;	w1 <<= 8;
	w1 |= *octets++;

	w0 = 0;
	w0 |= *octets++;	w0 <<= 8;
	w0 |= *octets++;	w0 <<= 8;
	w0 |= *octets++;	w0 <<= 8;
	w0 |= *octets++;

	x = w0;
	x = ldexp (x, -32);
	x += w1;
	x = ldexp (x, -32);

	return (x);
}

/*
 * Generate points in decimal mode.
 *
 * Note that 2**128 is a 39-digit integer, so we can obtain up to 38
 * random decimal digits from a single block of 128 random bits.
 * Let:
 *
 *	K = floor (38 / ndigits),
 *
 *	N = K * ndigits,
 *
 *	M = 10**N,
 *
 *	A = 2**128,
 *
 *	Q = floor (A / M),
 *
 *	R = A - Q*M.
 *
 * We throw away any random block whose value >= Q*M, since these
 * blocks cause values 0, 1, 2, ..., R-1 to occur Q+1 times, whereas
 * values R, R+1, ..., M-1 occur only Q times.  This discard occurs
 * with probability R/2**128.  This probability should always be < 1/2,
 * but if Q is too small, we decrease K by 1.  This gives us fewer digits
 * per block, but helps to assure that we do not get into a bad discard loop.
 */

	static
	void
gen_decimal_points (

struct my_data * mdp,		/* IN/OUT: random source */
int		n,		/* IN: number of points to generate */
int		ndigits,	/* IN: number of digits per coordinate */
int		nplaces,	/* IN: number of those digits that are */
				/*     fractional (i.e., lie to the */
				/*     right of the decimal point) */
FILE *		fp		/* IN/OUT: file to write into */
)
{
int		i;
int64u		x, y;
char		xbuf [32], ybuf [32];

	if (nplaces < 0) {
		nplaces = ndigits;
	}

	for (i = 0; i < n; i++) {
		x = get_decimal_number (&(mdp -> dbuf));
		y = get_decimal_number (&(mdp -> dbuf));

		convert_to_decimal (xbuf, x, ndigits, nplaces);
		convert_to_decimal (ybuf, y, ndigits, nplaces);

		fprintf (fp, "%s %s\n", xbuf, ybuf);
	}
}

/*
 * Generate the given number of points into the given array.
 */

	static
	int
generate_array (

struct PRNG_obj *	obj,		/* IN/OUT: our instance object */
double *		array,		/* IN/OUT: array to fill in */
int			n		/* IN: number of points to generate */
)
{
struct my_data *	mdp;

	FATAL_ERROR_IF ((obj EQ NULL) OR (array EQ NULL));

	mdp = (struct my_data *) (obj -> opaque);

	switch (mdp -> options.mode) {
	case MODE_BINARY:
		gen_binary_array (&(mdp -> prng), n, array);
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
 * Generate points in binary mode into given array. 
 */

	static
	void
gen_binary_array (

struct PRNG *	state,		/* IN/OUT: random source */
int		n,		/* IN: number of points to generate */
double *	array		/* IN/OUT: Array to fill in */
)
{
int		i;
int8u		buf [16];
double		x, y;

	/* We get 128-bit words from our random stream, each of	*/
	/* which gives us 2 64-bit mantissas, or one point!	*/

	for (i = 0; i < n; i++) {
		compute_next_random_block (buf, state);

		x = get_binary_double (buf);
		y = get_binary_double (buf + 8);

		*array++	= x;
		*array++	= y;
	}
}

/*
 * Generate points in decimal mode into given array.
 *
 * Note that 2**128 is a 39-digit integer, so we can obtain up to 38
 * random decimal digits from a single block of 128 random bits.
 * Let:
 *
 *	K = floor (38 / ndigits),
 *
 *	N = K * ndigits,
 *
 *	M = 10**N,
 *
 *	A = 2**128,
 *
 *	Q = floor (A / M),
 *
 *	R = A - Q*M.
 *
 * We throw away any random block whose value >= Q*M, since these
 * blocks cause values 0, 1, 2, ..., R-1 to occur Q+1 times, whereas
 * values R, R+1, ..., M-1 occur only Q times.  This discard occurs
 * with probability R/2**128.  This probability should always be < 1/2,
 * but if Q is too small, we decrease K by 1.  This gives us fewer digits
 * per block, but helps to assure that we do not get into a bad discard loop.
 */

	static
	void
gen_decimal_array (

struct my_data * mdp,		/* IN/OUT: random source */
int		n,		/* IN: number of points to generate */
int		ndigits,	/* IN: number of digits per coordinate */
int		nplaces,	/* IN: number of those digits that are */
				/*     fractional (i.e., lie to the */
				/*     right of the decimal point) */
double *	array		/* IN/OUT: array to fill in */
)
{
int		i;
double		x, y, scale;

	if (nplaces < 0) {
		nplaces = ndigits;
	}

	scale = 1.0;
	for (i = 0; i < nplaces; i++) {
		scale *= 10.0;
	}

	for (i = 0; i < n; i++) {
		x = (double) get_decimal_number (&(mdp -> dbuf));
		y = (double) get_decimal_number (&(mdp -> dbuf));

		x /= scale;	/* Rounding! */
		y /= scale;	/* Rounding! */

		*array++	= x;
		*array++	= y;
	}
}

/*
 * Initialize the given DecBuf object so that it produces a sequence
 * of uniformly-distributed random integers, each having D decimal
 * digits, where D=ndigits.
 */

	static
	void
init_dbuf (

struct DecBuf *	dp,		/* IN/OUT: DecBuf object to initialize */
struct PRNG *	state,		/* IN/OUT: random source */
int		ndigits		/* IN: number of digits per coordinate */
)
{
int		i, octet;
int		K, N;
mpz_t		A, M, Q, R;

	K = MAX_DIGITS_PER_BLOCK / ndigits;
	N = K * ndigits;

	mpz_init (A);
	mpz_init (M);
	mpz_init (Q);
	mpz_init (R);

	mpz_ui_pow_ui (A, 2, 128);	/* A = 2**128	*/
	mpz_ui_pow_ui (M, 10, N);	/* M = 10**N	*/
	mpz_tdiv_qr (Q, R, A, M);	/* A = Q*M + R	*/
	if (mpz_cmp_ui (Q, 4) < 0) {
		--K;
		N = K * ndigits;
		mpz_ui_pow_ui (M, 10, N);
		mpz_tdiv_qr (Q, R, A, M);
	}

	memset (dp, 0, sizeof (*dp));
	dp -> state	= state;
	dp -> ndigits	= ndigits;
	dp -> K		= K;
	dp -> N		= N;
	dp -> nbuf	= 0;

	/* Our discard limit is Q*M. */
	mpz_mul (A, Q, M);

	for (i = 15; i >= 0; i--) {
		octet = mpz_tdiv_qr_ui (A, R, A, 256);
		dp -> limit [i] = octet;
	}

	mpz_clear (R);
	mpz_clear (Q);
	mpz_clear (M);
	mpz_clear (A);
}

/*
 * Get the next random integer from the given DecBuf object.
 */

	static
	int64u
get_decimal_number (

struct DecBuf *	dp	/* IN/OUT: DecBuf object to get next number from */
)
{
int		nbuf;

	nbuf = dp -> nbuf;
	if (nbuf <= 0) {
		/* Need to get more numbers into the buffer. */
		refill_decimal_buffer (dp);
		nbuf = dp -> nbuf;
		FATAL_ERROR_IF (nbuf <= 0);
	}

	--nbuf;
	dp -> nbuf = nbuf;

	return (dp -> buf [nbuf]);
}

/*
 * Get the next block of random bits, carve it up into uniformly-distributed
 * integers having the correct number of decimal digits, and place these
 * integers into the buffer.
 */

	static
	void
refill_decimal_buffer (

struct DecBuf *	dp		/* IN/OUT: DecBuf object to refill */
)
{
int		i, k, cmp;
mpz_t		Q, R, M;
int64u		word0, word1;
int8u		block [16];

	for (;;) {
		/* Get the next block of random bits. */
		compute_next_random_block (block, dp -> state);

		/* See if random word is strictly less than our limit. */
		cmp = memcmp (block, dp -> limit, 16);
		if (cmp < 0) break;

		/* Need to discard this block and pick another. */
	}

	mpz_init (Q);
	mpz_init (R);
	mpz_init (M);
	for (i = 0; i < 16; i++) {
		mpz_mul_2exp (Q, Q, 8);
		mpz_add_ui (Q, Q, block [i]);
	}
	mpz_ui_pow_ui (M, 10, dp -> ndigits);

	k = dp -> K;
	for (i = k - 1; i >= 0; i--) {
		mpz_tdiv_qr (Q, R, Q, M);
		word0 = mpz_get_ui (R);
		mpz_tdiv_q_2exp (R, R, 32);
		word1 = mpz_get_ui (R);

		dp -> buf [i] = (word1 << 32) | word0;
	}
	dp -> nbuf = k;

	mpz_clear (M);
	mpz_clear (R);
	mpz_clear (Q);
}

/*
 * Convert the given int64u to an ASCII decimal number of exactly ndigits.
 */

	static
	void
convert_to_decimal (

char *		buf,		/* IN/OUT: buffer to receive decimal number */
int64u		value,		/* IN: number to convert */
int		ndigits,	/* IN: number of digits to produce */
int		nplaces		/* IN: number of those digits */
)
{
int	n1, n2;

	FATAL_ERROR_IF (ndigits <= 0);
	FATAL_ERROR_IF (nplaces > ndigits);

	if ((nplaces < 0) OR (nplaces EQ ndigits)) {
		/* Make all of the digits be fractional.	*/
		/* Format is "0.XXXXXXX".			*/
		buf [0] = '0';
		buf [1] = '.';
		value = put_dec_digs (&buf [2], value, ndigits);
		buf [2 + ndigits] = '\0';
	}
	else if (nplaces EQ 0) {
		/* A pure integer -- no decimal point. */
		value = put_dec_digs (buf, value, ndigits);
		buf [ndigits] = '\0';
		blank_out_leading_zeros (buf, ndigits - 1);
	}
	else {
		/* Some digits before decimal point, some after. */
		n1 = ndigits - nplaces;
		n2 = nplaces;
		value = put_dec_digs (&buf [n1 + 1], value, n2);
		buf [n1] = '.';
		value = put_dec_digs (&buf [0], value, n1);
		buf [n1 + 1 + n2] = '\0';
		blank_out_leading_zeros (buf, n1 - 1);
	}

	FATAL_ERROR_IF (value NE 0);
}

/*
 * Convert the N least-significant digits of the given value
 * into ASCII decimal digits.  Return the given value with
 * these digits stripped (i.e., value / 10**N).
 * NOTE: this routine does *NOT* terminate the generated string
 * with a '\0' byte!
 */

	static
	int64u
put_dec_digs (

char *		buf,		/* IN/OUT: buffer to receive digit string */
int64u		value,		/* IN: number to convert */
int		ndigits		/* IN: The number N of digits to convert */
)
{
int		digit;

	FATAL_ERROR_IF (ndigits <= 0);

	do {
		digit = value % 10;
		value /= 10;
		--ndigits;
		buf [ndigits] = '0' + digit;
	} while (ndigits > 0);

	return (value);
}

/*
 * Blank out up to N leading zeros in the given string.
 */

	static
	void
blank_out_leading_zeros (

char *		buf,		/* IN/OUT: string to process */
int		n		/* IN: max number of zeros to blank out */
)
{
	while (n > 0) {
		if (*buf NE '0') break;
		*buf++ = ' ';
		--n;
	}
}

/*
 * Compute the next block of 128 random bits, which are written
 * into the given 16-byte buffer.
 */

	static
	void
compute_next_random_block (

int8u *		buf,		/* IN/OUT: buffer to write random bytes into */
struct PRNG *	state		/* IN/OUT: PRNG state object */
)
{
int		i, octet;
mpz_t		q, r;
int8u		temp [16];

	/* Compute temp[] = LC(n) XOR IV. */
	mpz_init_set (q, state -> lc);
	mpz_init (r);
	for (i = 15; i >= 0; i--) {
		octet = mpz_tdiv_qr_ui (q, r, q, 256);
		temp [i] = octet ^ state -> iv [i];
	}
	mpz_clear (r);
	mpz_clear (q);

	/* Encrypt temp[] in ECB mode. */
	_gst_aes256_encrypt_ecb (&(state -> ctx), temp);

	memcpy (state -> iv, temp, 16);
	memcpy (buf, temp, 16);

	/* Update LC(n) state for next time. */
	mpz_add (state -> lc, state -> lc, state -> b);
	mpz_tdiv_r_2exp (state -> lc, state -> lc, 128);
	mpz_add_ui (state -> n, state -> n, 1);
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
struct PRNG *		state;

	FATAL_ERROR_IF (obj EQ NULL);

	mdp = (struct my_data *) (obj -> opaque);
	state = &(mdp -> prng);

#define WRITE_MPZ_HEX(fp, zval) \
	do { \
		mpz_out_str ((fp), 16, (zval)); \
		fprintf ((fp), "\n"); \
	} while (0)

	WRITE_MPZ_HEX (fp, state -> a);
	WRITE_MPZ_HEX (fp, state -> b);
	WRITE_MPZ_HEX (fp, state -> n);
	WRITE_MPZ_HEX (fp, state -> lc);

#undef WRITE_MPZ_HEX

	write_hex_octets (fp, state -> ctx.key, 32);
	write_hex_octets (fp, state -> ctx.encryption_key, 32);
	write_hex_octets (fp, state -> ctx.decryption_key, 32);

	write_hex_octets (fp, state -> iv, 16);

	if (ferror (fp)) {
		return (EIO);
	}

	return (0);
}

/*
 * Write the given sequence of octects to the given string as a
 * newline-terminated string of ASCII hex digits.
 */

	static
	void
write_hex_octets (

FILE *		fp,		/* IN/OUT - stream to write into */
const int8u *	octets,		/* IN - octet string to write */
int		n		/* IN - number of octets to write */
)
{
int		i, byte;

	static const char hex [16] = "0123456789ABCDEF";

	for (i = 0; i < n; i++) {
		byte = octets [i];
		fprintf (fp,
			 "%c%c",
			 hex [(byte >> 4) & 0xF],
			 hex [byte & 0xF]);
	}
	fprintf (fp, "\n");
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
char			c;
struct my_data *	mdp;
struct PRNG *		state;

	FATAL_ERROR_IF ((obj EQ NULL) OR (fp EQ NULL));

	mdp = (struct my_data *) (obj -> opaque);
	state = &(mdp -> prng);


#define READ_MPZ_HEX(zval, fp) \
	do { \
		mpz_inp_str ((zval), fp, 16); \
		c = fgetc (fp); \
		if (c NE '\n') return (FALSE); \
	} while (0)

	READ_MPZ_HEX (state -> a, fp);
	READ_MPZ_HEX (state -> b, fp);
	READ_MPZ_HEX (state -> n, fp);
	READ_MPZ_HEX (state -> lc, fp);

#undef READ_MPZ_HEX

	if (NOT read_hex_octets (fp, state -> ctx.key,    32)) return (FALSE);
	if (NOT read_hex_octets (fp, state -> ctx.encryption_key, 32)) return (FALSE);
	if (NOT read_hex_octets (fp, state -> ctx.decryption_key, 32)) return (FALSE);

	if (NOT read_hex_octets (fp, state -> iv, 16)) return (FALSE);

	c = fgetc (fp);
	if (c NE EOF) return (FALSE);

	/* The file appears to be consistent. */

	return (TRUE);
}

/*
 * Read a sequence of N octets from the given input stream.  The octets
 * are encoded as ASCII hex digits terminated by a newline.
 */

	static
	bool
read_hex_octets (

FILE *		fp,		/* IN: input stream to read from */
int8u *		octets,		/* IN/OUT: buffer to read octets into */
int		n		/* IN: number of octets to read */
)
{
int		i, c, dig1, dig2;

	for (i = 0; i < n; i++) {
		c = fgetc (fp);
		if (c < 0) return (FALSE);
		dig1 = hexdig (c);
		if (dig1 < 0) return (FALSE);

		c = fgetc (fp);
		if (c < 0) return (FALSE);
		dig2 = hexdig (c);
		if (dig2 < 0) return (FALSE);

		octets [i] = (dig1 << 4) | dig2;
	}
	c = fgetc (fp);
	if (c NE '\n') return (FALSE);

	/* The octet string appears to be consistent. */

	return (TRUE);
}

/*
 * Set the given mpz_t object to a value represented as a hex string.
 */

	static
	void
my_mpz_set_hex (

mpz_ptr		dest,
const char *	hexval
)
{
char		c;
int		dig;

	mpz_set_ui (dest, 0);
	for (;;) {
		c = *hexval++;
		if (c EQ '\0') break;
		dig = hexdig (c);
		FATAL_ERROR_IF (dig < 0);

		mpz_mul_2exp (dest, dest, 4);
		mpz_add_ui (dest, dest, dig);
	}
}

/*
 * Convert the given ASCII character to its hex digit value, or -1 if
 * the character is not valid ASCII hex.
 */

	static
	int
hexdig (

char		c
)
{
	/* This code assumes ASCII encoding! */
	if (c <  '0') return (-1);
	if (c <= '9') return (c - '0');
	if (c <  'A') return (-1);
	if (c <= 'F') return (c - 'A' + 10);
	if (c <  'a') return (-1);
	if (c <= 'f') return (c - 'a' + 10);
	return (-1);
}

#endif	/* HAVE_GMP */

/*
 * The public implementation object exposed by this implementation.
 */

const struct PRNG_imp	rand_points_PRNG_aes_256 = {
#if defined(HAVE_GMP)

	.available	= TRUE,
	.name		= "AES-256",
	.create		= create_instance,

#else

	.available	= FALSE,
	.name		= "AES-256",
	.create		= NULL,
#endif
};
