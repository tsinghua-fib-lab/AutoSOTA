/***********************************************************************

	$Id: rand_points.h,v 1.5 2023/04/03 17:09:05 warme Exp $

	File:	rand_points.h
	Rev:	e-3
	Date:	04/03/2023

	Copyright (c) 2016, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Interfaces between the "rand_points.c" program and a
	one of its Pseudo-Random Number Generator (PRNG)
	implementations.  This allows each PRNG to be implemented
	in a separate source file.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Created.
	e-2:	09/24/2016	warme
		: Reorganize include files.
	e-3:	04/03/2023	warme
		: Add interface to generate points into an array.

************************************************************************/

#ifndef RAND_POINTS_H
#define RAND_POINTS_H

#include <stdio.h>
#include "gsttypes.h"

struct PRNG_imp;
struct PRNG_obj;
struct PRNG_options;

enum Mode {
	MODE_DECIMAL,		/* Generate fixed number of decimal digits */
	MODE_BINARY,		/* Generate binary IEEE double values */
};

/*
 * This object encapsulates all of the options that control the
 * format of random points that are generated.
 */

struct PRNG_options {
	enum Mode	mode;		/* Mode (binary or decimal) */
	/* Used only for decimal mode: */
	int		ndigits;	/* Number of digits to generate */
	int		nplaces;	/* Number of those digits that	*/
					/* are fractional		*/
	/* Options controlling the seed: */
	bool		randomize;	/* Use "cur_time" field to randomize */
	int32u		cur_time;	/* Fold this value into the initial */
					/* seed to "randomize" it */
	const char *	key;		/* Key to use.  If non-NULL, this */
					/* string is folded into the	  */
					/* *default* seed.		  */
};

/*
 * Defines for each available implementation.
 */

#define	PRNG_IMP_LEGACY		0	/* Original (bad) generator */
#define	PRNG_IMP_NEW		1	/* The `new' generator (better) */
#define	PRNG_IMP_AES_256	2	/* Generator that uses AES-256. */
					/* This one is the best (so far) */

#define PRNG_NUM_IMPS		3	/* Nuber of implementations */

#define PRNG_VALID_IMP_NUMBER(_n) ((0 <= (_n)) AND ((_n) < PRNG_NUM_IMPS))

/*
 * Object that encapsulates all of the functionality that rand_points.c
 * needs from one of its PRNG implementations.
 */

struct PRNG_imp {
	/* Is this implementation even available? */
	bool	available;

	const char *	name;	/* Name of this implementation. */

	/* Create and return an instance of this PRNG implementation	*/
	/* having the given options.  Returns NULL (after printing	*/
	/* an error message) if this implementation does not like any	*/
	/* of the given options.					*/
	struct PRNG_obj *
		(*create) (const struct PRNG_options * opts);
};

#define PRNG_IMP_UNAVAILABLE(imp) \
	((NOT ((imp) -> available)) OR ((imp) -> create EQ NULL))

#define PRNG_IMP_AVAILABLE(imp)	(NOT (PRNG_IMP_UNAVAILABLE (imp)))

/*
 * An instance of a PRNG object.
 */

struct PRNG_obj {
	/* Type index of generator that created us. */
	int	type;

	/* Data known only to the implementation. */
	void *	opaque;

	/* Read the seed from the given input stream.  Returns */
	/* TRUE if success, FALSE otherwise. */
	bool	(*read_seed) (struct PRNG_obj * obj, FILE * fp);

	/* Write the given number of points out to the given stream. */
	int	(*gen_points) (struct PRNG_obj * obj, FILE * fp, int npoints);

	/* Fill in the given array with generated points. */
	int	(*gen_array) (struct PRNG_obj * obj, double * array, int npoints);

	/* Write the current seed to the given output stream. */
	int	(*write_seed) (struct PRNG_obj * obj, FILE * fp);

	/* Virtual destructor. */
	void	(*destruct) (struct PRNG_obj * obj);
};

/* Forward declarations for each implementation. */

extern const struct PRNG_imp		rand_points_PRNG_legacy;
extern const struct PRNG_imp		rand_points_PRNG_new;
extern const struct PRNG_imp		rand_points_PRNG_aes_256;

#endif
