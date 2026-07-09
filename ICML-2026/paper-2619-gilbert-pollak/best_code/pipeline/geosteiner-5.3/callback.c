/***********************************************************************

	$Id: callback.c,v 1.2 2022/11/19 13:45:50 warme Exp $

	File:	callback.c
	Rev:	e-1
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Stuff for invoking callback functions, and the various
	Node Accessor API functions.

************************************************************************

	Modification Log:

	e-1:	11/18/2022	warme
		: Created.

************************************************************************/

#include "callback.h"

#include "bb.h"
#include "fatal.h"
#include "geosteiner.h"
#include "logic.h"
#include "solver.h"
#include "steiner.h"
#include <string.h>

/*
 * Global Functions
 */

void	_gst_invoke_bb_callback (int			wherefrom,
				 struct bbnode *	node);

int	gst_node_gst_node_branch_direction (gst_node_ptr node_accessor);
int	gst_node_gst_node_branch_var (gst_node_ptr node_accessor);
int	gst_node_get_lb (gst_node_ptr	node_accessor,
			 double *	lb);
int	gst_node_get_lb_status (gst_node_ptr node_accessor);
int	gst_node_gst_lp_index (gst_node_ptr node_accessor);
int	gst_node_get_node_depth (gst_node_ptr node_accessor);
int	gst_node_get_node_index (gst_node_ptr node_accessor);
int	gst_node_get_parent_node_index (gst_node_ptr node_accessor);
int	gst_node_get_solution (gst_node_ptr	node_accessor,
			       double *		x);
gst_solver_ptr	gst_node_get_solver (gst_node_ptr node_accessor);
int	gst_node_get_ub (gst_node_ptr	node_accessor,
			 double *	ub);
double	gst_node_get_z (gst_node_ptr node_accessor);

/*
 * An object that is opaque to library clients.  It is a "Node Accessor"
 * object.  We do not want to directly give "struct bbnode" pointers to
 * user code.  When invoking callback functions, we instead create one
 * of these node accessor objects, and implement various API functions
 * that let the user query info from the node accessor object.
 */

struct gst_node {
	long int	node_ac_magic;	/* Magic number for validation */
	struct bbnode *	real_node;	/* Pointer to our real node. */
};

/* In hex, this is roughly "NODE AC NODE AC " */
#define NODE_AC_MAGIC	0x90DE0AC090DE0ACF

/*
 * Local Functions
 */

static struct bbnode *	get_real_node (gst_node_ptr accessor);

/*
 * Invoke the callback with given "wherefrom" value and the given
 * branch-and-bound node.
 */

	void
_gst_invoke_bb_callback (

int		wherefrom,	/* IN: Where are we being called from */
struct bbnode *	node		/* IN: branch-and-bound node */
)
{
struct bbinfo *		bbip;
gst_solver_ptr		solver;

	FATAL_ERROR_IF (node EQ NULL);

	bbip = node -> owner;

	FATAL_ERROR_IF (bbip EQ NULL);

	solver = bbip -> solver;

	FATAL_ERROR_IF (solver EQ NULL);

	if (solver -> bb_callback_func NE NULL) {
		/* Make local Node Accessor object. */
		struct gst_node		accessor;
		accessor.node_ac_magic	= NODE_AC_MAGIC;
		accessor.real_node	= node;

		/* Invoke the callback function. */
		solver -> bb_callback_func (wherefrom,
					    &accessor,
					    solver -> bb_callback_data);
	}
}

/*
 * Node accessor to get the solver.
 */

	gst_solver_ptr
gst_node_get_solver (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> owner -> solver);
}

/*
 * Node accessor to get the current nodes LP objective value.
 */

	double
gst_node_get_z (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> z);
}

/*
 * Node accessor to get the current nodes "lower bound status".
 */

	int
gst_node_get_lb_status (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
int		retval;
struct bbnode *	node;

	node = get_real_node (accessor);

	retval = _gst_get_lb_status (node);

	return (retval);
}

/*
 * Node accessor to get the index of the current node.
 */

	int
gst_node_get_node_index (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> num);
}

/*
 * Node accessor to get the index of the parent of the current node.
 */

	int
gst_node_get_parent_node_index (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> parent);
}

/*
 * Node accessor to get the depth of the current node.
 */

	int
gst_node_get_node_depth (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> depth);
}

/*
 * Node accessor to get the branch variable used to create the
 * current node from its parent.
 */

	int
gst_node_get_node_branch_var (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> var);
}

/*
 * Node accessor to get the direction in which the branch variable
 * was constrained to create the current node from its parent.
 */

	int
gst_node_get_node_branch_direction (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> dir);
}

/*
 * Node accessor to get the index of the LP most recently solved
 * at the current node.
 */

	int
gst_node_get_lp_index (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	node = get_real_node (accessor);

	return (node -> iter);
}

/*
 * Node accessor to get the solution vector.
 */

	int
gst_node_get_solution (

gst_node_ptr	accessor,	/* IN: Node Accessor object */
double *	x		/* IN/OUT: Solution vector */
)
{
int		ncols, retval;
struct bbnode *	node;

	node = get_real_node (accessor);

	ncols	= node -> owner -> cip -> num_edges;

	if ((ncols > 0) AND (x EQ NULL)) {
		retval = GST_ERR_UNDEFINED;
	}
	else {
		memcpy (x, node -> x, ncols * sizeof (x [0]));
		retval = 0;
	}

	return (retval);
}

/*
 * Node accessor to get the current node's lower bound vector.
 */

	int
gst_node_get_lb (

gst_node_ptr	accessor,	/* IN: Node Accessor object */
double *	lb		/* IN/OUT: Lower bound vector */
)
{
int		j, ncols, retval;
bitmap_t *	fixed;
bitmap_t *	value;
struct bbnode *	node;
double		lbval;

	node = get_real_node (accessor);

	ncols	= node -> owner -> cip -> num_edges;

	if ((ncols > 0) AND (lb EQ NULL)) {
		retval = GST_ERR_UNDEFINED;
	}
	else {
		fixed = node -> fixed;
		value = node -> value;
		for (j = 0; j < ncols; j++) {
			lbval = 0.0;
			if (BITON (fixed, j) AND BITON (value, j)) {
				lbval = 1.0;
			}
			lb [j] = lbval;
		}
		retval = 0;
	}

	return (retval);
}

/*
 * Node accessor to get the current node's upper bound vector.
 */

	int
gst_node_get_ub (

gst_node_ptr	accessor,	/* IN: Node Accessor object */
double *	ub		/* IN/OUT: Upper bound vector */
)
{
int		j, ncols, retval;
bitmap_t *	fixed;
bitmap_t *	value;
struct bbnode *	node;
double		ubval;

	node = get_real_node (accessor);

	ncols	= node -> owner -> cip -> num_edges;

	if ((ncols > 0) AND (ub EQ NULL)) {
		retval = GST_ERR_UNDEFINED;
	}
	else {
		fixed = node -> fixed;
		value = node -> value;
		for (j = 0; j < ncols; j++) {
			ubval = 1.0;
			if (BITON (fixed, j) AND (NOT BITON (value, j))) {
				ubval = 0.0;
			}
			ub [j] = ubval;
		}
		retval = 0;
	}

	return (retval);
}

/*
 * Validate the given Node Accessor and return the "real" node pointer.
 */

	static
	struct bbnode *
get_real_node (

gst_node_ptr	accessor	/* IN: Node Accessor object */
)
{
struct bbnode *	node;

	FATAL_ERROR_IF (accessor EQ NULL);
	FATAL_ERROR_IF (accessor -> node_ac_magic NE NODE_AC_MAGIC);

	node = accessor -> real_node;

	FATAL_ERROR_IF (node EQ NULL);

	return (node);
}
