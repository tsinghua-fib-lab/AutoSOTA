/***********************************************************************

	$Id: ckpt.c,v 1.18 2023/04/03 17:49:27 warme Exp $

	File:	ckpt.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1999, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for checkpointing and restarting the computation.

************************************************************************

	Modification Log:

	a-1:	03/08/99	warme
		: Created.
	b-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Uses parameters.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
		: Make features unconditional.
	e-4:	11/18/2022	warme
		: Initialize bbnode owner.
		: Fix -Wall issue.
	e-5:	04/03/2023	warme
		: Fix -Wall issue.

************************************************************************/

#include "ckpt.h"

#include "config.h"
#include "bb.h"
#include "bbsubs.h"
#include "constrnt.h"
#include "cputime.h"
#include <errno.h>
#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "parmblk.h"
#include "solver.h"
#include "steiner.h"
#include <string.h>
#include <unistd.h>

/*
 * Global Routines
 */

bool		_gst_checkpoint_needed (struct bbinfo * bbip);
void		_gst_merge_constraints (struct bbinfo * bbip, char ** paths);
struct bbinfo *	_gst_restore_checkpoint (struct gst_hypergraph * cip,
					 gst_param_ptr		 params);
double *	_gst_restore_upper_bound_checkpoint (struct bbinfo * bbip);
void		_gst_write_checkpoint (struct bbinfo * bbip);
void		_gst_write_upper_bound_checkpoint (struct bbinfo * bbip);


/*
 * Local Equates
 */

#define	MAGIC_NUMBER	0xC3CBD0D4	/* 'CKPT' with top bits on... */

#define	LATEST_CHECKPOINT_VERSION	1


/*
 * Local Types
 */

struct v0_bbnode {		/* bbnode in version 0 checkpoint files */
	double		z;
	bool		optimal;
	int		num;
	int		iter;
	int		parent;
	int		index [2];
	int		var;
	int		dir;
	int		depth;
	int		br1cnt;
	bitmap_t *	fixed;
	bitmap_t *	value;
	int		n_uids;
	int *		bc_uids;
	int *		bc_row;
	int *		rstat;
	int *		cstat;
	double *	xhist;
	int		xh_index;
	double *	bheur;
	struct bbnode *	next;
	struct bbnode *	prev;
};


/*
 * The version 0 bbnode kept this many previous LP solutions.
 */

#define	V0_NUM_X_HISTORIES_PER_NODE	8


/*
 * Local Routines
 */

static bool		check_error (FILE *, char *, gst_channel_ptr);
static int		get_int (FILE *);
static int		merge_cpool (struct bbinfo *, struct bbinfo *);
static FILE *		open_checkpoint_file (char *, char *, gst_param_ptr);
static void		put_int (int, FILE *);
static struct bbinfo *	read_bbinfo (FILE *, struct gst_hypergraph *, int);
static struct bbnode *	read_bbnode (FILE *, struct bbinfo *, int);
static bool		read_bbstats (FILE *, struct bbinfo *, int);
static bool		read_bbtree (FILE *, struct bbinfo *, int);
static bool		read_cpool (FILE *, struct bbinfo *, int);
static bool		read_header (FILE *, struct gst_hypergraph *, int *);
static struct bbnode *	read_v0_bbnode (FILE *, struct bbinfo *, int);
static void		write_bbinfo (FILE *,
				      struct bbinfo *,
				      int,
				      cpu_time_t);
static void		write_bbnode (FILE *,
				      struct bbnode *,
				      struct bbinfo *,
				      int);
static void		write_bbstats (FILE *, struct bbinfo *, int);
static void		write_bbtree (FILE *, struct bbinfo *, int);
static void		write_cpool (FILE *, struct bbinfo *, int);
static void		write_header (FILE *, struct bbinfo *, int);


/*
 * This routine determines if enough CPU time has passed so that a
 * checkpoint is required.
 */

	bool
_gst_checkpoint_needed (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
cpu_time_t		now;
double			interval;

	if (bbip -> params -> checkpoint_filename EQ NULL) return (FALSE);
	interval = bbip -> params -> checkpoint_interval;
	if (interval <= 0.0) return (FALSE);

	now = _gst_get_cpu_time ();
	if (bbip -> next_ckpt_time EQ 0) {
		/* Schedule first checkpoint... */
		bbip -> next_ckpt_time =
				now + _gst_double_seconds_to_cpu_time_t (interval);
		return (FALSE);
	}
	return (now >= (bbip -> next_ckpt_time));
}

/*
 * The main routine to write out a checkpoint file containing the current
 * state of the computation.
 */

	void
_gst_write_checkpoint (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int		len;
char *		fname;
char *		nname;
char *		tname;
const char *	checkpoint_filename;
FILE *		fp;
int		version;
cpu_time_t	t0;
cpu_time_t	t1;
gst_param_ptr	params;
gst_channel_ptr	trace;

	params = bbip -> params;
	checkpoint_filename = params -> checkpoint_filename;

	if (checkpoint_filename EQ NULL) return;

	/* Exclude checkpointing time from CPU figures. */
	t0 = _gst_get_cpu_time ();

	trace = params -> print_solve_trace;

	len = strlen (checkpoint_filename);

	fname = NEWA (len + 5, char);
	nname = NEWA (len + 5, char);
	tname = NEWA (len + 5, char);

	sprintf (fname, "%s.chk", checkpoint_filename);
	sprintf (nname, "%s.new", checkpoint_filename);
	sprintf (tname, "%s.tmp", checkpoint_filename);

	version = LATEST_CHECKPOINT_VERSION;

	do {
		fp = fopen (tname, "w");
		if (fp EQ NULL) {
			gst_channel_printf (trace,
				"_gst_write_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}

		write_header (fp, bbip, version);
		if (check_error (fp, tname, trace)) break;

		write_bbinfo (fp, bbip, version, t0);
		if (check_error (fp, tname, trace)) break;

		write_bbstats (fp, bbip, version);
		if (check_error (fp, tname, trace)) break;

		write_cpool (fp, bbip, version);
		if (check_error (fp, tname, trace)) break;

		write_bbtree (fp, bbip, version);
		if (check_error (fp, tname, trace)) break;

		if (fflush (fp) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}

#ifdef HAVE_FSYNC
		if (fsync (fileno (fp)) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}
#endif
		if (fclose (fp) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}
		fp = NULL;

		/* The file is complete once it is named foo.new (nname). */
		/* Anything named foo.tmp (tname) is assumed to be junk.  */

#ifdef HAVE_RENAME
		if (rename (tname, nname) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_checkpoint: %s - %s, %s\n",
				strerror (errno), tname, nname);
			break;
		}
#else
		unlink (nname);
		if (link (tname, nname) NE 0) {
			gst_channel_printf (trace,
				" %% _gst_write_checkpoint: %s - %s, %s\n",
				strerror (errno), tname, nname);
			break;
		}
		unlink (tname);
#endif
#ifdef HAVE_SYNC
		sync ();
#endif

		/* Now replace the previous checkpoint (if any) with	*/
		/* the latest.						*/

#ifdef HAVE_RENAME
		if (rename (nname, fname) NE 0) {
			gst_channel_printf (trace,
				" %% _gst_write_checkpoint: %s - %s, %s\n",
				strerror (errno), nname, fname);
			break;
		}
#else
		unlink (fname);
		if (link (nname, fname) NE 0) {
			gst_channel_printf (trace,
				" %% _gst_write_checkpoint: %s - %s, %s\n",
				strerror (errno), nname, fname);
			break;
		}
		unlink (nname);
#endif

#ifdef HAVE_SYNC
		sync ();
#endif

	} while (FALSE);

	t1 = _gst_get_cpu_time ();

	bbip -> next_ckpt_time =
		t1 + _gst_double_seconds_to_cpu_time_t (
				bbip -> params -> checkpoint_interval);

	/* Exclude checkpointing time from CPU figures. */
	bbip -> t0 += (t1 - t0);

	if (fp NE NULL) {
		fclose (fp);
	}

	free (tname);
	free (nname);
	free (fname);
}

/*
 * Attempt to restore a saved checkpoint.  Returns NULL if no checkpoint
 * file was specified, no saved checkpoint exists, or the checkpoint is
 * invalid in any way.
 */

	struct bbinfo *
_gst_restore_checkpoint (

struct gst_hypergraph *	cip,		/* IN - compatibility info */
gst_param_ptr		params
)
{
FILE *			fp;
struct bbinfo *		rval;
struct bbinfo *		bbip;
int			version;

	rval = NULL;	/* return value NULL until we are done. */
	bbip = NULL;	/* no partially constructed problem yet. */

	do {	/* Used only for "break"... */

		if (params -> checkpoint_filename EQ NULL) break;

		fp = open_checkpoint_file (".new", ".chk", params);
		if (fp EQ NULL) break;

		if (NOT read_header (fp, cip, &version)) break;
		if (feof (fp)) break;

		bbip = read_bbinfo (fp, cip, version);
		if (bbip EQ NULL) break;
		if (feof (fp)) break;

		if (NOT read_bbstats (fp, bbip, version)) break;
		if (feof (fp)) break;

		if (NOT read_cpool (fp, bbip, version)) break;
		if (feof (fp)) break;

		if (NOT read_bbtree (fp, bbip, version)) break;

		rval = bbip;

	} while (FALSE);

	if (rval NE NULL) {
		/* We successfully restored the checkpoint! */
		/* Set CPU "start" time to include time from previons runs. */
		rval -> t0 = _gst_get_cpu_time () - rval -> t0;

		return (rval);
	}

	/* Checkpoint restore failed.  Clean up any partially built stuff. */
	if (bbip NE NULL) {
		_gst_destroy_bbinfo (bbip);
	}

	return (NULL);
}

/*
 * This routine merges the constraints from one or more checkpoint
 * files into the constraint pool of the given bbinfo.
 */

	void
_gst_merge_constraints (

struct bbinfo *		bbip,		/* IN/OUT - bbinfo to merge into */
char **			paths		/* IN - checkpoint file pathnames */
)
{
int			n;
char *			path;
struct gst_hypergraph *	cip;
struct bbinfo *		bbip2;
int			version;
FILE *			fp;

	cip = bbip -> cip;

	for (;;) {
		path = *paths++;
		if (path EQ NULL) break;

		fp = fopen (path, "r");
		if (fp EQ NULL) {
			perror (path);
			exit (1);
		}
		bbip2 = NULL;
		n = -1;
		do {	/* Used only for "break"... */
			if (NOT read_header (fp, cip, &version)) break;
			if (feof (fp)) break;

			bbip2 = read_bbinfo (fp, cip, version);
			if (bbip2 EQ NULL) break;
			if (feof (fp)) break;

			if (NOT read_bbstats (fp, bbip2, version)) break;
			if (feof (fp)) break;

			if (NOT read_cpool (fp, bbip2, version)) break;
			if (feof (fp)) break;

			/* Constraint pool successfully read!  Merge! */
			n = merge_cpool (bbip, bbip2);
		} while (FALSE);

		if (bbip2 NE NULL) {
			_gst_destroy_bbinfo (bbip2);
		}

		fclose (fp);

		gst_channel_printf (bbip -> params -> print_solve_trace,
			"\nMerged %d constraints from %s\n\n", n, path);
	}
}

/*
 * Write out the checkpoint file header.
 */

	static
	void
write_header (

FILE *		fp,		/* IN - stream to write header to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to write */
)
{
int			i;
int			total_edge_cardinality;
int			nverts;
int			nedges;
struct gst_hypergraph *	cip;

	cip = bbip -> cip;

	nverts = cip -> num_verts;
	nedges = cip -> num_edges;

	total_edge_cardinality = 0;
	for (i = 0; i < nedges; i++) {
		total_edge_cardinality += cip -> edge_size [i];
	}

	put_int (MAGIC_NUMBER, fp);
	put_int (version, fp);
	put_int (nverts, fp);
	put_int (nedges, fp);
	put_int (total_edge_cardinality, fp);
}

/*
 * Read the header from the given input stream.
 */

	static
	bool
read_header (

FILE *			fp,		/* IN - stream to read from */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
int *			version		/* OUT - data version number */
)
{
int			i;
int			total_edge_cardinality;
int			nverts;
int			nedges;

	nverts = cip -> num_verts;
	nedges = cip -> num_edges;

	total_edge_cardinality = 0;
	for (i = 0; i < nedges; i++) {
		total_edge_cardinality += cip -> edge_size [i];
	}

	*version = 0;

	i = get_int (fp);
	if (i NE ((int) MAGIC_NUMBER)) return (FALSE);

	i = get_int (fp);
	if (i > LATEST_CHECKPOINT_VERSION) return (FALSE);
	*version = i;

	i = get_int (fp);
	if (i NE nverts) return (FALSE);

	i = get_int (fp);
	if (i NE nedges) return (FALSE);

	i = get_int (fp);
	if (i NE total_edge_cardinality) return (FALSE);

	/* Header passes all sanity checks... */

	return (TRUE);
}

/*
 * Write out the bbinfo.
 */

	static
	void
write_bbinfo (

FILE *		fp,		/* IN - stream to write bbinfo to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version,	/* IN - version of data to write */
cpu_time_t	tchk		/* IN - start of checkpoint CPU time */
)
{
cpu_time_t	t0_save;

	/* Temporarily make the "t0" field indicate the current	*/
	/* branch-and-cut CPU time used so far...		*/

	t0_save		= bbip -> t0;
	bbip -> t0	= tchk - t0_save;

	fwrite (bbip, 1, sizeof (*bbip), fp);

	bbip -> t0	= t0_save;
}

/*
 * Read in the bbinfo.  Zap all fields that will be rebuilt from
 * scratch, or from other places in the file.
 */

	static
	struct bbinfo *
read_bbinfo (

FILE *			fp,	/* IN - stream to write bbinfo to */
struct gst_hypergraph *	cip,	/* IN - compatibility info */
int			version	/* IN - version of data to write */
)
{
int			i;
int			n;
int			nedges;
int			nmasks;
struct bbinfo *		bbip;

	bbip = NEW (struct bbinfo);

	n = fread (bbip, 1, sizeof (*bbip), fp);
	if (n NE sizeof (*bbip)) {
		free ((char *) bbip);
		return (NULL);
	}

	nedges	= cip -> num_edges;
	nmasks	= cip -> num_edge_masks;

	/* Zap stuff that will be rebuilt from scratch, or read	*/
	/* in later...						*/

	bbip -> cip		= cip;
	bbip -> solver		= NULL;
	bbip -> vert_mask	= cip -> initial_vert_mask;
	bbip -> edge_mask	= cip -> initial_edge_mask;
	bbip -> lp		= NULL;
	bbip -> lpmem		= NULL;
	bbip -> cpool		= NULL;
	bbip -> bbtree		= NULL;
	bbip -> csip		= NULL;
	bbip -> best_z		= DBL_MAX;
	bbip -> _smt		= NULL;
	bbip -> node		= NULL;
	bbip -> _z		= 0.0;
	bbip -> _x		= NULL;
	bbip -> slack_size	= 0;
	bbip -> slack		= NULL;
	bbip -> dj		= NEWA (nedges, double);
	bbip -> fixed		= NEWA (nmasks, bitmap_t);
	bbip -> value		= NEWA (nmasks, bitmap_t);
	bbip -> statp		= NULL;
	bbip -> ubip		= NULL;
	bbip -> params		= NULL;
	bbip -> rcfile		= NULL;
	bbip -> failed_fcomps	= NULL;
	bbip -> next_ckpt_time	= 0;
	bbip -> force_branch_flag = FALSE;

	for (i = 0; i < nedges; i++) {
		bbip -> dj [i] = 0.0;
	}
	memset (bbip -> fixed, 0, nmasks * sizeof (bitmap_t));
	memset (bbip -> value, 0, nmasks * sizeof (bitmap_t));

	return (bbip);
}

/*
 * Write out the statistics.
 */

	static
	void
write_bbstats (

FILE *		fp,		/* IN - stream to statistics to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to write */
)
{
struct bbstats *	statp;

	statp = bbip -> statp;

	fwrite (statp, 1, sizeof (*statp), fp);
}

	static
	bool
read_bbstats (

FILE *		fp,		/* IN - stream to read from */
struct bbinfo *	bbip,		/* IN - branch-and-bound-info */
int		version		/* IN - version of data to write */
)
{
int			n;
struct bbstats *	statp;

	statp = NEW (struct bbstats);

	bbip -> statp = statp;

	n = fread (statp, 1, sizeof (*statp), fp);

	return (n EQ sizeof (*statp));
}

/*
 * Write out the entire constraint pool.
 */

	static
	void
write_cpool (

FILE *		fp,		/* IN - stream to write constraint pool to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to write */
)
{
int		i;
int		nrows;
struct cpool *	pool;
struct rcon *	rcp;
struct rcoef *	cp;

	pool	= bbip -> cpool;
	rcp	= pool -> rows;
	nrows	= pool -> nrows;

	/* Write the entire cpool structure. */
	fwrite (pool, 1, sizeof (*pool), fp);

	/* Write out the array of constraint info. */
	fwrite (rcp, 1, nrows * sizeof (*rcp), fp);

	/* For each constraint, write out the raw coefficients. */
	for (i = 0; i < nrows; i++) {
		cp = rcp -> coefs;
		fwrite (cp, 1, (rcp -> len + 1) * sizeof (*cp), fp);
		if (ferror (fp)) break;
		++rcp;
	}
}

/*
 * Read in the constraint pool.
 */

	static
	bool
read_cpool (

FILE *		fp,		/* IN - stream to read constraint pool from */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data being read */
)
{
int		i;
int		n;
int		nrows;
int		nzsize;
int		ncoefs;
struct cpool *	pool;
struct rcon *	rcp;
struct rcoef *	cp;
struct rblk *	blkp;
int *		hookp;

	pool	= NEW (struct cpool);

	/* Read in the entire cpool structure. */
	n = fread (pool, 1, sizeof (*pool), fp);
	if (n NE sizeof (*pool)) {
		free ((char *) pool);
		return (FALSE);
	}

	/* Allocate new row arrays... */
	rcp = NEWA (pool -> maxrows, struct rcon);
	pool -> rows	= rcp;
	pool -> lprows	= NEWA (pool -> maxrows, int);
	pool -> blocks	= NULL;
	pool -> cbuf	= NEWA (pool -> nvars + 1, struct rcoef);

	/* Zap the hash table. */
	for (i = 0; i < CPOOL_HASH_SIZE; i++) {
		pool -> hash [i] = -1;
	}

	/* Pool is now "consistent enough" to be freed	*/
	/* "normally", so remember it.			*/
	bbip -> cpool = pool;

	/* Read in the array of constraint row headers... */
	nrows	= pool -> nrows;
	n = fread (rcp, 1, nrows * sizeof (*rcp), fp);
	if ((n NE nrows * sizeof (*rcp)) OR feof (fp)) {
		return (FALSE);
	}

	/* Calculate the block size to use.  We let this be	*/
	/* 4 times the number of coefficients in the initial	*/
	/* rows.						*/

	nzsize = 0;
	for (i = 0; i < pool -> initrows; i++) {
		nzsize += (rcp [i].len + 1);
	}
	nzsize *= 4;

	/* Mark all rows as being NOT in the LP.			*/
	for (i = 0; i < pool -> nrows; i++) {
		rcp [i].lprow = -1;
	}

	/* Make the first row pending, so that we have SOMETHING in	*/
	/* the LP tableaux...						*/
	rcp [0].lprow		= -2;	/* row 0 is pending... */
	pool -> lprows [0]	= 0;
	pool -> npend		= 1;

	/* Allocate the first block. */
	blkp		= NEW (struct rblk);
	blkp -> next	= NULL;
	blkp -> base	= NEWA (nzsize, struct rcoef);
	blkp -> ptr	= blkp -> base;
	blkp -> nfree	= nzsize;
	pool -> blocks	= blkp;

	/* Read each constraint in, one at a time. */
	for (i = 0; i < nrows; i++) {
		ncoefs = rcp -> len + 1;
		if (blkp -> nfree < ncoefs) {
			blkp = NEW (struct rblk);
			blkp -> next	= pool -> blocks;
			blkp -> base	= NEWA (nzsize, struct rcoef);
			blkp -> ptr	= blkp -> base;
			blkp -> nfree	= nzsize;
			pool -> blocks = blkp;
		}
		cp = blkp -> ptr;
		blkp -> ptr	+= ncoefs;
		blkp -> nfree	-= ncoefs;

		n = fread (cp, 1, ncoefs * sizeof (*cp), fp);
		if (n NE ncoefs * sizeof (*cp)) break;
		if (feof (fp)) break;

		rcp -> coefs = cp;

		/* Add row back into hash table. */
		hookp = &(pool -> hash [rcp -> hval]);
		rcp -> next = *hookp;
		*hookp = i;

		++rcp;
	}

	return (i >= nrows);
}

/*
 * Write out the entire branch-and-bound tree.
 */

	static
	void
write_bbtree (

FILE *		fp,		/* IN - stream to write nodes to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to write */
)
{
int			num_nodes;
struct bbtree *		tp;
struct bbnode *		nodep;


	tp = bbip -> bbtree;

	num_nodes = 0;
	for (nodep = tp -> first; nodep NE NULL; nodep = nodep -> next) {
		++num_nodes;
	}

	FATAL_ERROR_IF ((tp -> heap [BEST_NODE_HEAP].nheap NE num_nodes) OR
			(tp -> heap [WORST_NODE_HEAP].nheap NE num_nodes));

	put_int (num_nodes, fp);
	put_int (tp -> snum, fp);
	put_int (tp -> node_policy, fp);

	for (nodep = tp -> first; nodep NE NULL; nodep = nodep -> next) {
		write_bbnode (fp, nodep, bbip, version);
	}
}

/*
 * Read in the entire branch-and-bound tree.
 */

	static
	bool
read_bbtree (

FILE *		fp,		/* IN - stream to read nodes from */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data being read */
)
{
int			i;
int			nmasks;
int			num_nodes;
struct bbtree *		tp;
struct bbnode *		nodep;

	nmasks = bbip -> cip -> num_edge_masks;

	tp = _gst_create_bbtree (nmasks);

	bbip -> bbtree = tp;

	num_nodes		= get_int (fp);
	tp -> snum		= get_int (fp);
	tp -> node_policy	= get_int (fp);

	for (i = 0; i < num_nodes; i++) {
		if (feof (fp)) return (FALSE);
		nodep = read_bbnode (fp, bbip, version);
		if (nodep EQ NULL) return (FALSE);

		_gst_append_node_to_tree (nodep, tp);
	}

	return (TRUE);
}

/*
 * Write a single branch-and-bound node to the given checkpoint file.
 */

	static
	void
write_bbnode (

FILE *		fp,		/* IN - stream to write nodes to */
struct bbnode *	nodep,		/* IN - node to write out */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to write */
)
{
int			nedges;
int			nmasks;
struct gst_hypergraph *	cip;

	cip = bbip -> cip;
	nedges = cip -> num_edges;
	nmasks = cip -> num_edge_masks;

	do {	/* Only for "breaking". */
		/* First, write out the basic node structure. */
		fwrite (nodep, 1, sizeof (*nodep), fp);

		if (ferror (fp)) break;

		/* Now write out the various sub-arrays. */
		fwrite (nodep -> x, 1, nedges * sizeof (double), fp);

		if (ferror (fp)) break;

		fwrite (nodep -> zlb, 1, nedges * (2 * sizeof (double)), fp);

		if (ferror (fp)) break;

		fwrite (nodep -> fixed, 1, nmasks * sizeof (bitmap_t), fp);

		if (ferror (fp)) break;

		fwrite (nodep -> value, 1, nmasks * sizeof (bitmap_t), fp);

		if (ferror (fp)) break;

		fwrite (nodep -> bc_uids,
			1,
			nodep -> n_uids * sizeof (nodep -> bc_uids [0]),
			fp);

		if (ferror (fp)) break;

		fwrite (nodep -> bc_row,
			1,
			nodep -> n_uids * sizeof (nodep -> bc_row [0]),
			fp);

		if (ferror (fp)) break;

		fwrite (nodep -> rstat,
			1,
			nodep -> n_uids * sizeof (nodep -> rstat [0]),
			fp);

		if (ferror (fp)) break;

		fwrite (nodep -> cstat,
			1,
			nedges * sizeof (nodep -> cstat [0]),
			fp);

		if (ferror (fp)) break;

		fwrite (nodep -> bheur,
			1,
			nedges * sizeof (nodep -> bheur [0]),
			fp);

	} while (FALSE);
}

/*
 * Read in a single branch-and-bound node from the given checkpoint file.
 */

	static
	struct bbnode *
read_bbnode (

FILE *		fp,		/* IN - stream to write nodes to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to write */
)
{
int			n;
int			nmasks;
int			ncols;
struct bbnode *		nodep;
struct bbnode *		rval;
struct gst_hypergraph *	cip;

	if (version <= 0) {
		nodep = read_v0_bbnode (fp, bbip, version);
		return (nodep);
	}

	cip = bbip -> cip;
	nmasks = cip -> num_edge_masks;

	ncols = cip -> num_edges;

	rval = NULL;

	do {	/* Only for "breaking". */
		/* First, read in the basic node structure. */
		nodep = NEW (struct bbnode);
		n = fread (nodep, 1, sizeof (*nodep), fp);
		if (n NE sizeof (*nodep)) break;
		if (ferror (fp)) break;

		/* Force in the proper owner. */
		nodep -> owner = bbip;

		/* Allocate the various sub-arrays. */
		nodep -> x		= NEWA (ncols, double);
		nodep -> zlb		= NEWA (2 * ncols, double);
		nodep -> fixed		= NEWA (nmasks, bitmap_t);
		nodep -> value		= NEWA (nmasks, bitmap_t);
		nodep -> bc_uids	= NEWA (nodep -> n_uids, int);
		nodep -> bc_row		= NEWA (nodep -> n_uids, int);
		nodep -> rstat		= NEWA (nodep -> n_uids, int);
		nodep -> cstat		= NEWA (ncols, int);
		nodep -> bheur		= NEWA (ncols, double);
		nodep -> next		= NULL;
		nodep -> prev		= NULL;

		/* Now read in the various sub-arrays. */
		n = fread (nodep -> x, 1, ncols * sizeof (double), fp);
		if (n NE ncols * sizeof (double)) break;
		if (ferror (fp)) break;

		n = fread (nodep -> zlb, 1, ncols * (2 * sizeof (double)), fp);
		if (n NE ncols * (2 * sizeof (double))) break;
		if (ferror (fp)) break;

		n = fread (nodep -> fixed, 1, nmasks * sizeof (bitmap_t), fp);
		if (n NE nmasks * sizeof (bitmap_t)) break;
		if (ferror (fp)) break;

		n = fread (nodep -> value, 1, nmasks * sizeof (bitmap_t), fp);
		if (n NE nmasks * sizeof (bitmap_t)) break;
		if (ferror (fp)) break;

		n = fread (nodep -> bc_uids,
			   1,
			   nodep -> n_uids * sizeof (nodep -> bc_uids [0]),
			   fp);
		if (n NE nodep -> n_uids * sizeof (nodep -> bc_uids [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> bc_row,
			   1,
			   nodep -> n_uids * sizeof (nodep -> bc_row [0]),
			   fp);
		if (n NE nodep -> n_uids * sizeof (nodep -> bc_row [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> rstat,
			   1,
			   nodep -> n_uids * sizeof (nodep -> rstat [0]),
			   fp);
		if (n NE nodep -> n_uids * sizeof (nodep -> rstat [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> cstat,
			   1,
			   ncols * sizeof (nodep -> cstat [0]),
			   fp);
		if (n NE ncols * sizeof (nodep -> cstat [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> bheur,
			   1,
			   ncols * sizeof (nodep -> bheur [0]),
			   fp);
		if (n NE ncols * sizeof (nodep -> bheur [0])) break;

		/* Success!  Now let the return-value be non-NULL. */
		rval = nodep;

	} while (FALSE);

	if (rval EQ NULL) {
		free ((char *) (nodep -> bheur));
		free ((char *) (nodep -> cstat));
		free ((char *) (nodep -> rstat));
		free ((char *) (nodep -> bc_row));
		free ((char *) (nodep -> bc_uids));
		free ((char *) (nodep -> value));
		free ((char *) (nodep -> fixed));
		free ((char *) (nodep -> zlb));
		free ((char *) (nodep -> x));
		free ((char *) nodep);
	}

	return (rval);
}

/*
 * Read in a single version 0 bbnode from the given checkpoint file, and
 * convert it to the current format.
 */

	static
	struct bbnode *
read_v0_bbnode (

FILE *		fp,		/* IN - stream to write nodes to */
struct bbinfo *	bbip,		/* IN - branch-and-bound info */
int		version		/* IN - version of data to read */
)
{
int			i;
int			j;
int			n;
int			nmasks;
int			ncols;
int			nhist;
int			xh_index;
struct bbnode *		nodep;
struct bbnode *		rval;
struct gst_hypergraph *	cip;
double *		xhist;
double *		bheur;
double *		x0;
double *		x1;
struct v0_bbnode	buf;

	cip = bbip -> cip;
	nmasks = cip -> num_edge_masks;

	ncols = cip -> num_edges;

	nhist = V0_NUM_X_HISTORIES_PER_NODE * ncols;
	xhist = NEWA (nhist, double);

	rval = NULL;

	do {	/* Only for "breaking". */
		nodep = NEW (struct bbnode);

		memset (nodep, 0, sizeof (*nodep));

		/* First, read in the basic node structure. */

		n = fread (&buf, 1, sizeof (buf), fp);
		if (n NE sizeof (buf)) break;
		if (ferror (fp)) break;

		/* Force in the proper owner. */
		nodep -> owner = bbip;

		/* Allocate the various sub-arrays. */
		nodep -> x		= NEWA (ncols, double);
		nodep -> zlb		= NEWA (2 * ncols, double);
		nodep -> fixed		= NEWA (nmasks, bitmap_t);
		nodep -> value		= NEWA (nmasks, bitmap_t);
		nodep -> bc_uids	= NEWA (buf.n_uids, int);
		nodep -> bc_row		= NEWA (buf.n_uids, int);
		nodep -> rstat		= NEWA (buf.n_uids, int);
		nodep -> cstat		= NEWA (ncols, int);
		nodep -> bheur		= NEWA (ncols, double);
		nodep -> next		= NULL;
		nodep -> prev		= NULL;

		/* Copy simple data. */
		nodep -> z		= buf.z;
		nodep -> optimal	= buf.optimal;
		nodep -> num		= buf.num;
		nodep -> iter		= buf.iter;
		nodep -> parent		= buf.parent;
		for (i = 0; i < NUM_BB_HEAPS; i++) {
			nodep -> index [i] = buf.index [i];
		}
		nodep -> var		= buf.var;
		nodep -> dir		= buf.dir;
		nodep -> depth		= buf.depth;
		nodep -> br1cnt		= buf.br1cnt;
		nodep -> cpiter		= -1;	/* Force LP to be re-solved. */
		nodep -> n_uids		= buf.n_uids;

		xh_index		= buf.xh_index;

		/* Now read in the various sub-arrays. */
		n = fread (nodep -> fixed, 1, nmasks * sizeof (bitmap_t), fp);
		if (n NE nmasks * sizeof (bitmap_t)) break;
		if (ferror (fp)) break;

		n = fread (nodep -> value, 1, nmasks * sizeof (bitmap_t), fp);
		if (n NE nmasks * sizeof (bitmap_t)) break;
		if (ferror (fp)) break;

		n = fread (nodep -> bc_uids,
			   1,
			   nodep -> n_uids * sizeof (nodep -> bc_uids [0]),
			   fp);
		if (n NE nodep -> n_uids * sizeof (nodep -> bc_uids [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> bc_row,
			   1,
			   nodep -> n_uids * sizeof (nodep -> bc_row [0]),
			   fp);
		if (n NE nodep -> n_uids * sizeof (nodep -> bc_row [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> rstat,
			   1,
			   nodep -> n_uids * sizeof (nodep -> rstat [0]),
			   fp);
		if (n NE nodep -> n_uids * sizeof (nodep -> rstat [0])) break;
		if (ferror (fp)) break;

		n = fread (nodep -> cstat,
			   1,
			   ncols * sizeof (nodep -> cstat [0]),
			   fp);
		if (n NE ncols * sizeof (nodep -> cstat [0])) break;
		if (ferror (fp)) break;

		n = fread (xhist, 1, nhist * sizeof (*xhist), fp);
		if (n NE nhist * sizeof (*xhist)) break;
		if (ferror (fp)) break;

		bheur = nodep -> bheur;
		n = fread (bheur, 1, ncols * sizeof (bheur [0]), fp);
		if (n NE ncols * sizeof (bheur [0])) break;

		/* Grab a copy of the most recent "x" from the obsolete	*/
		/* xhist, and compute bheur the new way from the xhist.	*/

		--xh_index;
		if (xh_index < 0) {
			xh_index += V0_NUM_X_HISTORIES_PER_NODE;
		}
		memcpy (nodep -> x,
			&xhist [xh_index * ncols],
			ncols * sizeof (double));

		for (i = 0; i < ncols; i++) {
			bheur [i] = 0.0;
		}

		xh_index = buf.xh_index;
		x0 = &xhist [xh_index * ncols];		/* oldest x */
		xh_index = (xh_index + 1) % V0_NUM_X_HISTORIES_PER_NODE;
		x1 = &xhist [xh_index * ncols];		/* second oldest x */

		for (i = 1; i < V0_NUM_X_HISTORIES_PER_NODE; i++) {
			for (j = 0; j < ncols; j++) {
				bheur [j] = 0.75 * bheur [j] + fabs (x1 [j] - x0 [j]);
			}
			x0 = x1;
			xh_index = (xh_index + 1) % V0_NUM_X_HISTORIES_PER_NODE;
			x1 = &xhist [xh_index * ncols];
		}

		/* Set the Z lower bounds to innocuous values. */
		for (i = 0; i < 2 * ncols; i++) {
			nodep -> zlb [i] = buf.z;
		}

		/* Success!  Now let the return-value be non-NULL. */
		rval = nodep;

	} while (FALSE);

	free ((char *) xhist);

	if (rval EQ NULL) {
		free ((char *) (nodep -> bheur));
		free ((char *) (nodep -> cstat));
		free ((char *) (nodep -> rstat));
		free ((char *) (nodep -> bc_row));
		free ((char *) (nodep -> bc_uids));
		free ((char *) (nodep -> value));
		free ((char *) (nodep -> fixed));
		free ((char *) (nodep -> zlb));
		free ((char *) (nodep -> x));
		free ((char *) nodep);
	}

	return (rval);
}

/*
 * This routine checks the given file stream for errors.  It prints an
 * error message and returns TRUE if an error has occurred.
 */

	static
	bool
check_error (

FILE *		fp,		/* IN - stream to check */
char *		fname,		/* IN - file name stream refers to */
gst_channel_ptr	trace
)
{
	if (ferror (fp)) {
		gst_channel_printf (trace,
			"Checkpoint write error: %s: %s\n",
			strerror (errno), fname);
		return (TRUE);
	}

	return (FALSE);
}
/*
 * The main routine to write out a checkpoint file containing the current
 * upper bound of the computation.
 */

	void
_gst_write_upper_bound_checkpoint (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int		i;
int		nedges;
int		len;
char *		fname;
char *		nname;
char *		tname;
char *		checkpoint_filename;
FILE *		fp;
bitmap_t *	smt;
cpu_time_t	t0;
cpu_time_t	t1;
gst_param_ptr	params;
gst_channel_ptr	trace;

	params = bbip -> params;
	checkpoint_filename = params -> checkpoint_filename;

	if (checkpoint_filename EQ NULL) return;

	/* Exclude checkpointing time from CPU figures. */
	t0 = _gst_get_cpu_time ();

	trace = params -> print_solve_trace;

	len = strlen (checkpoint_filename);

	fname = NEWA (len + 5, char);
	nname = NEWA (len + 5, char);
	tname = NEWA (len + 5, char);

	sprintf (fname, "%s.ub", checkpoint_filename);
	sprintf (nname, "%s.nub", checkpoint_filename);
	sprintf (tname, "%s.tmp", checkpoint_filename);

	nedges	= bbip -> cip -> num_edges;
	smt	= bbip -> solver -> solutions [0].edge_mask;

	do {
		fp = fopen (tname, "w");
		if (fp EQ NULL) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}

		for (i = 0; i < nedges; i++) {
			if (NOT BITON (smt, i)) continue;
			fprintf (fp, "%d\n", i);
		}

		if (fflush (fp) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}

#ifdef HAVE_FSYNC
		if (fsync (fileno (fp)) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}
#endif
		if (fclose (fp) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s\n",
				strerror (errno), tname);
			break;
		}
		fp = NULL;

		/* The file is complete once it is named foo.nub (nname). */
		/* Anything named foo.tmp (tname) is assumed to be junk.  */

#ifdef HAVE_RENAME
		if (rename (tname, nname) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s, %s\n",
				strerror (errno), tname, nname);
			break;
		}
#else
		unlink (nname);
		if (link (tname, nname) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s, %s\n",
				strerror (errno), tname, nname);
			break;
		}
		unlink (tname);
#endif
#ifdef HAVE_SYNC
		sync ();
#endif

		/* Now replace the previous checkpoint (if any) with	*/
		/* the latest.						*/

#ifdef HAVE_RENAME
		if (rename (nname, fname) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s, %s\n",
				strerror (errno), nname, fname);
			break;
		}
#else
		unlink (fname);
		if (link (nname, fname) NE 0) {
			gst_channel_printf (trace,
				"_gst_write_upper_bound_checkpoint: %s - %s, %s\n",
				strerror (errno), nname, fname);
			break;
		}
		unlink (nname);
#endif

#ifdef HAVE_SYNC
		sync ();
#endif

	} while (FALSE);

	t1 = _gst_get_cpu_time ();

	/* Exclude checkpointing time from CPU figures. */
	bbip -> t0 += (t1 - t0);

	if (fp NE NULL) {
		fclose (fp);
	}

	free (tname);
	free (nname);
	free (fname);
}

/*
 * Attempt to restore an upper bound saved in its checkpoint file.
 */

	double *
_gst_restore_upper_bound_checkpoint (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			nedges;
int			edge;
FILE *			fp;
double *		x;
struct gst_hypergraph *	cip;
cpu_time_t		t0;
cpu_time_t		t1;
bool			valid;

	t0 = _gst_get_cpu_time ();

	cip	= bbip -> cip;
	nedges	= cip -> num_edges;

	x = NULL;

	do {	/* Used only for "break"... */

		if (bbip -> params -> checkpoint_filename EQ NULL) break;

		fp = open_checkpoint_file (".nub", ".ub", bbip -> params);
		if (fp EQ NULL) break;

		x = NEWA (nedges, double);

		for (i = 0; i < nedges; i++) {
			x [i] = 0.0;
		}

		valid = TRUE;
		for (;;) {
			i = fscanf (fp, " %d", &edge);
			if (i < 1) break;
			if ((edge < 0) OR (edge >= nedges)) {
				valid = FALSE;
				break;
			}
			x [edge] = 1.0;
		}

		if (NOT valid) {
			free ((char *) x);
			x = NULL;
			break;
		}
	} while (FALSE);

	t1 = _gst_get_cpu_time ();

	/* Omit checkpoint restore time from CPU total. */
	bbip -> t0 += (t1 - t0);

	return (x);
}

/*
 * Write a single integer out to the given stream.
 */

	static
	void
put_int (

int		word,		/* IN - integer to write to stream */
FILE *		fp		/* IN - stream to write to */
)
{
	fwrite (&word, 1, sizeof (word), fp);
}


/*
 * Read a single integer from the given stream.
 */

	static
	int
get_int (

FILE *		fp		/* IN - stream to read from */
)
{
int		word;

	word = 0;

	fread (&word, 1, sizeof (word), fp);

	return (word);
}

/*
 * Attempt to open a checkpoint file.  If a file ending with suffix1
 * exists, it is the latest checkpoint file -- it was about to be renamed
 * with suffix2 but the operation was never completed.  We complete the
 * operation right now by renaming the file.
 * In either case, we then try to open the file named with suffix2.
 */

	static
	FILE *
open_checkpoint_file (

char *		suffix1,		/* IN - first suffix to try */
char *		suffix2,		/* IN - second suffix to try */
gst_param_ptr	params
)
{
int		len1;
int		len2;
char *		fname1;
char *		fname2;
char *		checkpoint_filename;
FILE *		fp;

	checkpoint_filename = params -> checkpoint_filename;

	len1 = strlen (checkpoint_filename);
	len2 = strlen (suffix1);
	fname1 = NEWA (len1 + len2 + 1, char);
	fname2 = NEWA (len1 + len2 + 1, char);

	strcpy (&fname1 [0], checkpoint_filename);
	strcpy (&fname2 [0], checkpoint_filename);
	strcpy (&fname1 [len1], suffix1);
	strcpy (&fname2 [len1], suffix2);

#ifdef HAVE_RENAME
	/* Rename the file.  We don't really care if it worked or not! */
	rename (fname1, fname2);
#else
	/* Here we assume that no other processes are messing with	*/
	/* the checkpoint file(s)...					*/
	fp = fopen (fname1, "r");
	if (fp EQ NULL) {
		gst_channel_ptr	trace;
		trace = params -> print_solve_trace;
		/* A nearly-complete checkpoint exists!  Finish the	*/
		/* job right now...					*/
		unlink (fname2);
		if (link (fname1, fname2) NE 0) {
			fclose (fp);
			gst_channel_printf (trace,
				"open_checkpoint_file: %s - %s, %s\n",
				strerror (errno), fname1, fname2);
			free (fname2);
			free (fname1);
			return (NULL);
		}
		fclose (fp);
		unlink (fname1);
	}
#endif

	fp = fopen (fname2, "r");

	free (fname2);
	free (fname1);

	return (fp);
}

/*
 * This routine merges the constraints from bbip2 -> cpool into the
 * constraint pool bbip1 -> cpool.
 */

	static
	int
merge_cpool (

struct bbinfo *		bbip1,		/* IN/OUT - receive merged pools */
struct bbinfo *		bbip2		/* IN - merge from here */
)
{
int			i;
int			n;
int			nadded;
struct cpool *		pool1;
struct cpool *		pool2;
struct rcon *		rcp;

	pool1	= bbip1 -> cpool;
	pool2	= bbip2 -> cpool;

	n = pool2 -> nrows;

	nadded = 0;

	for (i = 0; i < n; i++) {
		rcp = &(pool2 -> rows [i]);
		if (_gst_add_constraint_to_pool (pool1, rcp -> coefs, FALSE)) {
			++nadded;
		}
	}

	return (nadded);
}
