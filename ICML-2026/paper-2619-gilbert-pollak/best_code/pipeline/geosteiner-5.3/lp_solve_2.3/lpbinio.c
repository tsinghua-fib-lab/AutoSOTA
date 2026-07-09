/***********************************************************************

	File:	lpbinio.c
	Rev:	e-3
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution 4.0 International
	License.

************************************************************************

	Routines to load and dump LP problems in binary form so
	that numeric accuracy is not lost.

************************************************************************

	Modification Log:

	a-1:	09/03/96	warme
		: Created.
	b-1:	02/28/2001	warme
		: Include string.h for strcpy.
	e-1:	04/14/2015	warme
		: Changes for geosteiner 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	11/18/2022	warme
		: Fix -Wall issue.

************************************************************************/

#include "lpkit.h"
#include <string.h>


/*
 * Global Routines
 */

void		dump_lp (lprec * lp, char * filename);
lprec *		load_lp (FILE * stream);


/*
 * Local Equates
 */

#define	MOST_SIGNIFICANT_WORD_FIRST	1


/*
 * Local Routines
 */

static unsigned long	get4 (FILE *);
static double		getd (FILE *);
static void		put4 (FILE *, unsigned long);
static void		putd (FILE *, double);

/*
 * This routine dumps the given LP out to the given file in a binary
 * (big-endian) format that is transportable across machines (assuming
 * everybody uses 64-bit IEEE floating point, that is...).
 */

	void
dump_lp (

lprec *		lp,		/* IN - LP to dump */
char *		fname		/* IN - file name to dump to */
)
{
int		i;
FILE *		fp;

	fp = fopen (fname, "wb");
	if (fp == NULL) {
		perror ("dump_lp");
		return;
	}

	/* First write all of the scalar values... */

	put4 (fp, lp -> verbose);
	put4 (fp, lp -> print_duals);
	put4 (fp, lp -> print_sol);
	put4 (fp, lp -> debug);
	put4 (fp, lp -> print_at_invert);
	put4 (fp, lp -> trace);
	put4 (fp, lp -> anti_degen);
	put4 (fp, lp -> rows);
	put4 (fp, lp -> columns);

	/* don't bother with the name stuff... */

	put4 (fp, lp -> non_zeros);
	put4 (fp, lp -> basis_valid);

	/* don't bother to save eta either... */

	put4 (fp, lp -> num_inv);
	put4 (fp, lp -> max_num_inv);
	put4 (fp, lp -> bb_rule);
	put4 (fp, lp -> break_at_int);
	putd (fp, lp -> break_value);
	putd (fp, lp -> obj_bound);
	put4 (fp, lp -> iter);
	put4 (fp, lp -> total_iter);
	put4 (fp, lp -> max_level);
	put4 (fp, lp -> total_nodes);
	put4 (fp, lp -> maximise);
	put4 (fp, lp -> floor_first);
	put4 (fp, lp -> scaling_used);
	put4 (fp, lp -> columns_scaled);

	/* don't save the lagrange stuff... */

	put4 (fp, lp -> valid);
	putd (fp, lp -> infinite);
	putd (fp, lp -> epsilon);
	putd (fp, lp -> epsb);
	putd (fp, lp -> epsd);
	putd (fp, lp -> epsel);

	/* Now write all of the vector quantities... */

	for (i = 0; i < lp -> non_zeros; i++) {
		putd (fp, lp -> mat [i].value);
	}
	for (i = 0; i < lp -> non_zeros; i++) {
		put4 (fp, lp -> mat [i].row_nr);
	}
	for (i = 0; i <= lp -> columns; i++) {
		put4 (fp, lp -> col_end [i]);
	}
	for (i = 0; i < lp -> non_zeros; i++) {
		put4 (fp, lp -> col_no [i]);
	}
	for (i = 0; i <= lp -> rows; i++) {
		put4 (fp, lp -> row_end [i]);
	}
	for (i = 0; i <= lp -> rows; i++) {
		putd (fp, lp -> orig_rh [i]);
	}
	/* rh and rhs are temporaries that we need not save... */
	for (i = 0; i <= lp -> sum; i++) {
		put4 (fp, lp -> must_be_int [i]);
	}
	for (i = 0; i <= lp -> sum; i++) {
		putd (fp, lp -> orig_lowbo [i]);
		putd (fp, lp -> orig_upbo [i]);
	}
	for (i = 0; i <= lp -> rows; i++) {
		put4 (fp, lp -> bas [i]);
	}
	for (i = 0; i <= lp -> sum; i++) {
		put4 (fp, lp -> basis [i]);
	}
	for (i = 0; i <= lp -> sum; i++) {
		put4 (fp, lp -> lower [i]);
	}
	for (i = 0; i <= lp -> sum; i++) {
		putd (fp, lp -> solution [i]);
	}
	for (i = 0; i <= lp -> sum; i++) {
		putd (fp, lp -> best_solution [i]);
	}
	for (i = 0; i <= lp -> rows; i++) {
		putd (fp, lp -> duals [i]);
	}
	for (i = 0; i <= lp -> rows; i++) {
		put4 (fp, lp -> ch_sign [i]);
	}
	if (lp -> scaling_used) {
		for (i = 0; i <= lp -> sum; i++) {
			putd (fp, lp -> scale [i]);
		}
	}

	/* don't save the lagrange stuff... */

	fclose (fp);
}

/*
 * This routine loads in the LP problem from the given input stream.
 * The stream is assumed to contain the binary (big-endian) format
 * that is transportable across machines (assuming everybody uses
 * 64-bit IEEE floating point, that is...).
 */

	lprec *
load_lp (

FILE *		fp		/* IN - input stream to load LP from */
)
{
int		i;
int		rows;
int		cols;
int		sum;
lprec *		lp;

	CALLOC (lp, 1);

	/* First read all of the scalar values... */

	strcpy (lp -> lp_name, "unnamed");

	lp -> verbose		= get4 (fp);
	lp -> print_duals	= get4 (fp);
	lp -> print_sol		= get4 (fp);
	lp -> debug		= get4 (fp);
	lp -> print_at_invert	= get4 (fp);
	lp -> trace		= get4 (fp);
	lp -> anti_degen	= get4 (fp);
	lp -> rows		= get4 (fp);
	lp -> columns		= get4 (fp);

	rows = lp -> rows;
	cols = lp -> columns;
	sum = rows + cols;

	lp -> sum		= sum;
	lp -> sum_alloc		= sum;
	lp -> rows_alloc	= rows;
	lp -> columns_alloc	= cols;
	lp -> names_used	= FALSE;

	/* don't bother with the name stuff... */

	lp -> non_zeros		= get4 (fp);
	lp -> mat_alloc		= lp -> non_zeros;
	lp -> basis_valid	= get4 (fp);

	/* don't bother to load eta either... */

	lp -> num_inv		= get4 (fp);
	lp -> max_num_inv	= get4 (fp);
	lp -> bb_rule		= get4 (fp);
	lp -> break_at_int	= get4 (fp);
	lp -> break_value	= getd (fp);
	lp -> obj_bound		= getd (fp);
	lp -> iter		= get4 (fp);
	lp -> total_iter	= get4 (fp);
	lp -> max_level		= get4 (fp);
	lp -> total_nodes	= get4 (fp);
	lp -> maximise		= get4 (fp);
	lp -> floor_first	= get4 (fp);
	lp -> scaling_used	= get4 (fp);
	lp -> columns_scaled	= get4 (fp);

	/* don't load the lagrange stuff... */

	lp -> valid		= get4 (fp);
	lp -> infinite		= getd (fp);
	lp -> epsilon		= getd (fp);
	lp -> epsb		= getd (fp);
	lp -> epsd		= getd (fp);
	lp -> epsel		= getd (fp);

	CALLOC (lp -> mat, lp -> mat_alloc);
	CALLOC (lp -> col_no, lp -> mat_alloc);
	CALLOC (lp -> col_end, cols + 1);
	CALLOC (lp -> row_end, rows + 1);
	lp -> row_end_valid = FALSE;
	CALLOC (lp -> orig_rh, rows + 1);
	CALLOC (lp -> rh, rows + 1);
	CALLOC (lp -> rhs, rows + 1);
	CALLOC (lp -> must_be_int, sum + 1);
	CALLOC (lp -> orig_upbo, sum + 1);
	CALLOC (lp -> upbo, sum + 1);
	CALLOC (lp -> orig_lowbo, sum + 1);
	CALLOC (lp -> lowbo, sum + 1);
	CALLOC (lp -> bas, rows + 1);
	CALLOC (lp -> basis, sum + 1);
	CALLOC (lp -> lower, sum + 1);

	lp -> eta_valid = FALSE;
	lp -> eta_size = 0;
	lp -> eta_alloc = 10000;
	CALLOC (lp -> eta_value, lp -> eta_alloc);
	CALLOC (lp -> eta_row_nr, lp -> eta_alloc);
	CALLOC (lp -> eta_col_end, rows + lp -> max_num_inv);

	CALLOC (lp -> solution, sum + 1);
	CALLOC (lp -> best_solution, sum + 1);
	CALLOC (lp -> duals, rows + 1);
	CALLOC (lp -> ch_sign, rows + 1);
	if (lp -> scaling_used) {
		CALLOC (lp -> scale, sum + 1);
	}

	/* Now read all of the vector quantities... */

	for (i = 0; i < lp -> non_zeros; i++) {
		lp -> mat [i].value		= getd (fp);
	}
	for (i = 0; i < lp -> non_zeros; i++) {
		lp -> mat [i].row_nr		= get4 (fp);
	}
	for (i = 0; i <= lp -> columns; i++) {
		lp -> col_end [i]		= get4 (fp);
	}
	for (i = 0; i < lp -> non_zeros; i++) {
		lp -> col_no [i]		= get4 (fp);
	}
	for (i = 0; i <= lp -> rows; i++) {
		lp -> row_end [i]		= get4 (fp);
	}
	for (i = 0; i <= lp -> rows; i++) {
		lp -> orig_rh [i]		= getd (fp);
	}
	/* rh and rhs are temporaries that we need not save... */
	for (i = 0; i <= lp -> sum; i++) {
		lp -> must_be_int [i]		= get4 (fp);
	}
	for (i = 0; i <= lp -> sum; i++) {
		lp -> orig_lowbo [i]		= getd (fp);
		lp -> orig_upbo [i]		= getd (fp);
	}
	for (i = 0; i <= lp -> rows; i++) {
		lp -> bas [i]		= get4 (fp);
	}
	for (i = 0; i <= lp -> sum; i++) {
		lp -> basis [i]		= get4 (fp);
	}
	for (i = 0; i <= lp -> sum; i++) {
		lp -> lower [i]		= get4 (fp);
	}
	for (i = 0; i <= lp -> sum; i++) {
		lp -> solution [i]		= getd (fp);
	}
	for (i = 0; i <= lp -> sum; i++) {
		lp -> best_solution [i]		= getd (fp);
	}
	for (i = 0; i <= lp -> rows; i++) {
		lp -> duals [i]		= getd (fp);
	}
	for (i = 0; i <= lp -> rows; i++) {
		lp -> ch_sign [i]		= get4 (fp);
	}
	if (lp -> scaling_used) {
		for (i = 0; i <= lp -> sum; i++) {
			lp -> scale [i]		= getd (fp);
		}
	}

	/* don't load the lagrange stuff... */

	return (lp);
}

/*
 * This routine writes a 4-byte integer in big-endian byte order.
 */

	static
	void
put4 (

FILE *		fp,
unsigned long	data
)
{
char *		p;
char		buf [4];

	p = &buf [4];
	*--p = data;		data >>= 8;
	*--p = data;		data >>= 8;
	*--p = data;		data >>= 8;
	*--p = data;

	putc (*p++, fp);
	putc (*p++, fp);
	putc (*p++, fp);
	putc (*p++, fp);
}


/*
 * This routine writes an IEEE double in big-endian order.
 */

	static
	void
putd (

FILE *		fp,
double		data
)
{
unsigned long *	p;

	p = (unsigned long *) &data;

#if MOST_SIGNIFICANT_WORD_FIRST
	/* most-significant word first */
	put4 (fp, p [0]);
	put4 (fp, p [1]);
#else
	/* least-significant word first */
	put4 (fp, p [1]);
	put4 (fp, p [0]);
#endif
}

/*
 * This routine reads a 4-byte integer in big-endian byte order.
 */

	static
	unsigned long
get4 (

FILE *		fp
)
{
unsigned long	data;

	data = 0;
	data += getc (fp);	data <<= 8;
	data += getc (fp);	data <<= 8;
	data += getc (fp);	data <<= 8;
	data += getc (fp);

	return (data);
}


/*
 * This routine reads an IEEE double in big-endian order.
 */

	static
	double
getd (

FILE *		fp
)
{
union {
	unsigned int	word [2];
	double		fval;
} u;

#if MOST_SIGNIFICANT_WORD_FIRST
	/* most-significant word first */
	u.word [0] = get4 (fp);
	u.word [1] = get4 (fp);
#else
	/* least-significant word first */
	u.word [1] = get4 (fp);
	u.word [0] = get4 (fp);
#endif

	return (u.fval);
}
