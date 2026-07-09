/***********************************************************************

	$Id: parms.c,v 1.39 2022/11/19 13:45:53 warme Exp $

	File:	parms.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	The parameter system. Creation/deletion/queries/get/set.

************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "parms.h"

#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include <limits.h>
#include "logic.h"
#include "memory.h"
#include "parmblk.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>
#include "utils.h"

/*
 * Global Routines
 */

gst_param_ptr	gst_create_param (int *);
int		gst_free_param (gst_param_ptr);
int		gst_copy_param (gst_param_ptr, gst_param_ptr);
int		gst_get_param_type (int, int *);
int		gst_get_param_id (const char *, int *);
int		gst_set_param (gst_param_ptr, const char *, const char *);

int		gst_set_chn_param (gst_param_ptr, int,
				   gst_channel_ptr);
int		gst_get_chn_param (gst_param_ptr, int,
				   gst_channel_ptr *);
int		gst_set_dbl_param (gst_param_ptr, int, double);
int		gst_get_dbl_param (gst_param_ptr, int, double *);
int		gst_query_dbl_param (gst_param_ptr,
				     int, double *, double *,
				     double *, double *);
int		gst_set_int_param (gst_param_ptr, int, int);
int		gst_get_int_param (gst_param_ptr, int, int *);
int		gst_query_int_param (gst_param_ptr,
				     int, int *, int *, int *, int *);
int		gst_set_str_param (gst_param_ptr, int, const char *);
int		gst_get_str_param (gst_param_ptr, int, int *, char *);

void		_gst_initialize_parameters (struct environment *);
void		_gst_shutdown_parameters (struct gst_parmdefs *);

/*
 * Local Types
 */

struct intparm {
	int		index;
	size_t		offset;
	int		min_value;
	int		max_value;
	int		default_value;
};

struct dblparm {
	int		index;
	size_t		offset;
	double		min_value;
	double		max_value;
	double		default_value;
};

struct strparm {
	int		index;
	size_t		offset;
	const char *	default_value;
	bool		(*validate) (const char *);
};

struct chnparm {
	int		index;
	size_t		offset;
	bool		(*validate) (gst_channel_ptr);
};

/* This strucure permits us to handle parameters IDs that are	*/
/* not sequentially numbered.  It contains arrays of pointers	*/
/* to the various parameter definitions.  These pointers are	*/
/* allowed to be NULL for parameter IDs that don't exist.	*/

struct gst_parmdefs {
	const struct intparm **	iparms;
	int			niparms;
	const struct dblparm **	dparms;
	int			ndparms;
	const struct strparm **	sparms;
	int			nsparms;
	const struct chnparm **	cparms;
	int			ncparms;
};

#define INTROW(sym,val,var,min,max,dflt) \
	{val, offsetof(struct gst_param, var), min, max, dflt},

#define DBLROW(sym,val,var,min,max,dflt) \
	{val, offsetof(struct gst_param, var), min, max, dflt},

#define STRROW(sym,val,var,func,dflt) \
	{val, offsetof(struct gst_param, var), dflt, func},

#define CHNROW(sym,val,var,func) \
	{val, offsetof(struct gst_param, var)},

static const struct intparm intparm_table [] = {
	INTPARMS(INTROW)
};

static const struct dblparm dblparm_table [] = {
	DBLPARMS(DBLROW)
};

static const struct strparm strparm_table [] = {
	STRPARMS(STRROW)
};

static const struct chnparm chnparm_table [] = {
	CHNPARMS(CHNROW)
};

#undef INTROW
#undef DBLROW
#undef STRROW
#undef CHNROW

#define INTROW(sym,val,var,min,max,dflt) dflt,
#define DBLROW(sym,val,var,min,max,dflt) dflt,
#define STRROW(sym,val,var,func,dflt) dflt,
#define CHNROW(sym,val,var,func) NULL,

#define GST___PARMBLK_MAGIC 42

const struct gst_param	_gst_default_parmblk = {
	GST___PARMBLK_MAGIC,
	0,
	DBLPARMS(DBLROW)
	INTPARMS(INTROW)
	STRPARMS(STRROW)
	CHNPARMS(CHNROW)
};

#undef INTROW
#undef DBLROW
#undef STRROW
#undef CHNROW

/*
 * Local Macros
 */

#define PARAM_SLOT(p, offset, type) \
	(*((type *) ( ((char *) (p)) + (offset) )))

#define VALID_PARAM_SET(p) \
	(((p) NE NULL) AND ((p) -> magic EQ GST___PARMBLK_MAGIC))

#define PARAM_ID_IN_RANGE(i, min, maxfield) \
	(((i) >= min) AND ((i-min) <= (gst_env -> parmdefs -> maxfield)))

#define VALIDATE(var, p, i, min, ptrfield, maxfield)		\
	if (NOT VALID_PARAM_SET (p)) {				\
		res = GST_ERR_INVALID_PARAMETERS_OBJECT;	\
		break;						\
	}							\
	if (NOT PARAM_ID_IN_RANGE (i, min, maxfield)) {		\
		res = GST_ERR_UNKNOWN_PARAMETER_ID;		\
		break;						\
	}							\
	var = gst_env -> parmdefs -> ptrfield [i % 1000];	\
	if (var EQ NULL) {					\
		res = GST_ERR_UNKNOWN_PARAMETER_ID;		\
		break;						\
	}							\
	if (var -> index NE i) {				\
		FATAL_ERROR;					\
	}

#define VALIDATE_INT(var, p, i)	VALIDATE(var, p, i, 1000, iparms, niparms)
#define VALIDATE_DBL(var, p, i)	VALIDATE(var, p, i, 2000, dparms, ndparms)
#define VALIDATE_STR(var, p, i)	VALIDATE(var, p, i, 3000, sparms, nsparms)
#define VALIDATE_CHN(var, p, i)	VALIDATE(var, p, i, 4000, cparms, ncparms)

#define INTROW(sym,val,var,min,max,dflt) {#var, val},
#define DBLROW(sym,val,var,min,max,dflt) {#var, val},
#define STRROW(sym,val,var,func,dflt) {#var, val},
#define CHNROW(sym,val,var,func) {#var, val},

struct parmdef {
	const char *	symbol;
	int		id;
};

static const struct parmdef parmtable [] = {
	INTPARMS(INTROW)
	DBLPARMS(DBLROW)
	STRPARMS(STRROW)
	CHNPARMS(CHNROW)
	{NULL, 0},
};

#undef INTROW
#undef DBLROW
#undef STRROW
#undef CHNROW

/*
 * Initialize the environment to contain pointers to our various
 * parameter definitions.  These structures allow us to have
 * parameter IDs that are out of order, or have gaps in them.
 * This feature is useful for conditional compilation that includes
 * or removes various features, while retaining fixed parameter ID
 * numbers for ABI compatibility purposes.
 */

	void
_gst_initialize_parameters (

struct environment *	env	/* IN/OUT - GeoSteiner environment */
)
{
int			i;
int			j;
int			max;
struct gst_parmdefs *	pdefs;

	FATAL_ERROR_IF ((env EQ NULL) OR (env -> parmdefs NE NULL));

	pdefs = NEW (struct gst_parmdefs);
	memset (pdefs, 0, sizeof (*pdefs));

#define NELTS(array)	(sizeof (array) / sizeof (array [0]))
#define PROCESS(TABLE, N, TYPE, TYPESTR, PVAR, NVAR)			\
	max = -1;							\
	for (i = 0; i < NELTS (TABLE); i++) {				\
		j = TABLE [i].index;					\
		FATAL_ERROR_IF ((j / 1000) NE N);			\
		j %= 1000;						\
		if (j > max) {						\
			max = j;					\
		}							\
	}								\
	if (max >= 0) {							\
		pdefs -> PVAR = NEWA (max + 1, TYPE *);			\
		memset (pdefs -> PVAR, 0, (max + 1) * sizeof (TYPE *));	\
	}								\
	pdefs -> NVAR = max;						\
	for (i = 0; i < NELTS (TABLE); i++) {				\
		j = TABLE [i].index % 1000;				\
		FATAL_ERROR_IF (pdefs -> PVAR [j] NE NULL);		\
		pdefs -> PVAR [j] = &TABLE [i];				\
	}								\

	PROCESS (intparm_table, 1, const struct intparm, "int",
		 iparms, niparms);
	PROCESS (dblparm_table, 2, const struct dblparm, "double",
		 dparms, ndparms);
	PROCESS (strparm_table, 3, const struct strparm, "string",
		 sparms, nsparms);
	PROCESS (chnparm_table, 4, const struct chnparm, "channel",
		 cparms, ncparms);

	env -> parmdefs = pdefs;
}

/*
 * Free up the arrays we allocated to hold param definitions.
 */

	void
_gst_shutdown_parameters (

struct gst_parmdefs *	pdefs	/* IN - parameter defs to shutdown */
)
{
	if (pdefs -> iparms NE NULL) {
		free (pdefs -> iparms);
	}
	if (pdefs -> dparms NE NULL) {
		free (pdefs -> dparms);
	}
	if (pdefs -> sparms NE NULL) {
		free (pdefs -> sparms);
	}
	if (pdefs -> cparms NE NULL) {
		free (pdefs -> cparms);
	}
	free (pdefs);
}

/*
 * Allocates space for a parameters structure and copies the default.
 */

	gst_param_ptr
gst_create_param (

int *		status
)
{
gst_param_ptr	tmp;

	GST_PRELUDE

	tmp = NEW (struct gst_param);
	*tmp = _gst_default_parmblk;

	if (status NE NULL) {
		*status = 0;
	}

	GST_POSTLUDE

	return (tmp);
}

/*
 * Free the space taken by a parameters structure.
 */

	int
gst_free_param (

gst_param_ptr		param
)
{
int			res;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		if ((param EQ NULL) OR (param EQ &_gst_default_parmblk)) {
			/* Do nothing and exit quietly. */
			break;
		}
		if (NOT VALID_PARAM_SET (param)) {
			res = GST_ERR_INVALID_PARAMETERS_OBJECT;
			break;
		}

		/* Free up all string parameters... */
#define STRROW(sym,val,var,func,dflt) \
		if (param -> var NE NULL) { \
			free (param -> var); \
			param -> var = NULL; \
		}
		STRPARMS (STRROW)
#undef STRROW

		/* NULL out any channels that might be present. */
#define CHNROW(sym,val,var,func) param -> var = NULL;
		CHNPARMS (CHNROW)
#undef CHNROW

		param->magic = 0;
		free (param);
	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Copy the contents of an existing parameter set. If NULL is given as the
 * source then the destination is overwritten with the default parameter set.
 */

	int
gst_copy_param (

gst_param_ptr		dest,
gst_param_ptr		src
)
{
int			res;

	GST_PRELUDE

	res = 0;

	if (src EQ NULL) {
		src = (gst_param_ptr) &_gst_default_parmblk;
	}

	do {		/* Used only for "break"... */

		if (NOT VALID_PARAM_SET (dest)) {
			res = GST_ERR_INVALID_PARAMETERS_OBJECT;
			break;
		}

		if (NOT VALID_PARAM_SET (src)) {
			res = GST_ERR_INVALID_PARAMETERS_OBJECT;
			break;
		}

		*dest = *src;

		/* Must make copies of all string parameters! */

#define STRROW(sym,val,var,func,dflt) dest -> var = _gst_strdup (src -> var);
		STRPARMS (STRROW)
#undef STRROW
	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Return the type of the given parameter.
 */

	int
gst_get_param_type (

int			whichparam,	/* IN - parameter id */
int *			type		/* OUT - the type of the parameter */
)
{
int		res;
int		tval;

	GST_PRELUDE

	res = 0;
	tval = -1;

#define VALIDATE_PARAMETER(i, min, ptrfield, maxfield) \
		if (NOT PARAM_ID_IN_RANGE (i, min, maxfield) OR \
		    (gst_env -> parmdefs -> ptrfield [i % 1000] EQ NULL)) { \
			/* Invalid parameter number, or parameter is of wrong type. */ \
			res = GST_ERR_UNKNOWN_PARAMETER_ID; \
		}

	switch (whichparam / 1000) {
	case 1:
		VALIDATE_PARAMETER (whichparam, 1000, iparms, niparms);
		tval = GST_PARAMTYPE_INTEGER;
		break;
	case 2:
		VALIDATE_PARAMETER (whichparam, 2000, dparms, ndparms);
		tval = GST_PARAMTYPE_DOUBLE;
		break;
	case 3:
		VALIDATE_PARAMETER (whichparam, 3000, sparms, nsparms);
		tval = GST_PARAMTYPE_STRING;
		break;
	case 4:
		VALIDATE_PARAMETER (whichparam, 4000, cparms, ncparms);
		tval = GST_PARAMTYPE_CHANNEL;
		break;
	default:
		res = GST_ERR_UNKNOWN_PARAMETER_ID;
		break;
	}

#undef VALIDATE_PARAMETER

	if (type NE NULL) {
		*type = tval;
	}

	GST_POSTLUDE

	return (res);
}

/*
 * Return the type of the given parameter.
 */

	int
gst_get_param_id (

const char *		param_name,	/* IN - parameter name */
int *			param_id	/* OUT - the id of the parameter */
)
{
int			res;
int			p_id;
const struct parmdef *	p;

	GST_PRELUDE

	res = 0;
	p_id = -1;

	do {		/* Used only for "break"... */

		/* Skip the prefix if it is there */
		if (strncasecmp (param_name, "GST_PARAM_", 10) EQ 0) {
			param_name += 10;
		}

		res = GST_ERR_UNKNOWN_PARAMETER_ID;
		for (p = &parmtable [0]; p -> symbol NE NULL; p++) {
			if (strcasecmp (p -> symbol, param_name) EQ 0) {
				p_id = p -> id;
				res = 0;
				break;
			}
		}
	} while (FALSE);

	if (param_id NE NULL) {
		*param_id = p_id;
	}

	GST_POSTLUDE

	return (res);
}

/*
 * Set the value of any parameter which is not a channel. The function takes
 * two strings. One being the string name of the parameter and one being
 * its value also given as a string. The purpose of this function is to make
 * it easy to parse command line arguments for the parameters.
 *
 * Note that it should not be used in general since a linear search is done
 * for each parameter.
 */

	int
gst_set_param (

gst_param_ptr	params,		/* IN/OUT - parameter set */
const char *	name,		/* IN - parameter name */
const char *	value		/* IN - parameter value */
)
{
int		res;
int		id;
int		type;

	GST_PRELUDE

	do {		/* Used only for "break"... */
		res = gst_get_param_id (name, &id);
		if (res NE 0) {
			break;
		}

		res = gst_get_param_type (id, &type);
		if (res NE 0) {
			break;
		}

		switch (type) {
		case GST_PARAMTYPE_INTEGER:
			res = gst_set_int_param (params, id, atoi(value));
			break;

		case GST_PARAMTYPE_DOUBLE:
			res = gst_set_dbl_param (params, id, atof(value));
			break;

		case GST_PARAMTYPE_STRING:
			res = gst_set_str_param (params, id, value);
			break;

		case GST_PARAMTYPE_CHANNEL:
			res = GST_ERR_INVALID_PARAMETER_TYPE;
			break;

		default:
			/* Should never get here. */
			FATAL_ERROR;
		}
	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Set the value of a 'double' parameter
 */

	int
gst_set_dbl_param (

gst_param_ptr		param,
int			whichparam,
double			newvalue
)
{
int			res;
const struct dblparm *	parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_DBL (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if ((newvalue < parm->min_value) OR
		    (newvalue > parm->max_value)) {
			/* Parameter value out of range. */
			res = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
			break;
		}

		PARAM_SLOT (param, parm->offset, double) = newvalue;

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Get the value from a 'double' parameter
 */

	int
gst_get_dbl_param (

gst_param_ptr		param,
int			whichparam,
double*			value
)
{
int			res;
const struct dblparm *	parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_DBL (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if (value EQ NULL) break;

		*value = PARAM_SLOT (param, parm->offset, double);

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Query information about a 'double' parameter
 */

	int
gst_query_dbl_param (

gst_param_ptr		param,
int			whichparam,
double *		current_value,
double *		default_value,
double *		min_value,
double *		max_value
)
{
int			res;
const struct dblparm *	parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_DBL (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if (current_value NE NULL) {
			*current_value = PARAM_SLOT (param, parm->offset, double);
		}
		if (default_value NE NULL) {
			*default_value = parm->default_value;
		}
		if (min_value NE NULL) {
			*min_value = parm->min_value;
		}
		if (max_value NE NULL) {
			*max_value = parm->max_value;
		}
	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Set the value of an 'int' parameter
 */

	int
gst_set_int_param (

gst_param_ptr		param,
int			whichparam,
int			newvalue
)
{
int			res;
const struct intparm	*parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_INT (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if ((newvalue < parm -> min_value) OR
		    (newvalue > parm->max_value)) {
			/* Parameter value out of range. */
			res = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
		}

		PARAM_SLOT (param, parm->offset, int) = newvalue;

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Get the value from an 'int' parameter
 */

	int
gst_get_int_param (

gst_param_ptr		param,
int			whichparam,
int *			value
)
{
int			res;
const struct intparm	*parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_INT (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if (value EQ NULL) break;

		*value = PARAM_SLOT (param, parm->offset, int);

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Query information about an 'int' parameter
 */

	int
gst_query_int_param (

gst_param_ptr		param,
int			whichparam,
int *			current_value,
int *			default_value,
int *			min_value,
int *			max_value
)
{
int			res;
const struct intparm *	parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_INT (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if (current_value NE NULL) {
			*current_value = PARAM_SLOT (param, parm->offset, int);
		}
		if (default_value NE NULL) {
			*default_value = parm->default_value;
		}
		if (min_value NE NULL) {
			*min_value = parm->min_value;
		}
		if (max_value NE NULL) {
			*max_value = parm->max_value;
		}
	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Set the channel to be used by a 'channel' parameter
 */

	int
gst_set_chn_param (

gst_param_ptr		param,
int			whichparam,
gst_channel_ptr		chan
)
{
int			res;
const struct chnparm *	parm;
bool			(*funcp) (gst_channel_ptr);

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_CHN (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		funcp = parm -> validate;
		if ((funcp NE NULL) AND (NOT (*funcp) (chan))) {
			res = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
			break;
		}

		PARAM_SLOT (param, parm->offset, gst_channel_ptr) = chan;

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Get the channel currently used by a 'channel' parameter
 */

	int
gst_get_chn_param (

gst_param_ptr		param,
int			whichparam,
gst_channel_ptr *	chan
)
{
int			res;
const struct chnparm *	parm;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_CHN (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		if (chan EQ NULL) break;

		*chan = PARAM_SLOT (param, parm->offset, gst_channel_ptr);

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Set the string for a 'string' parameter
 */

	int
gst_set_str_param (

gst_param_ptr		param,
int			whichparam,
const char *		string
)
{
int			res;
const struct strparm *	parm;
char **			hookp;
bool			(*funcp) (const char *);

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_STR (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		funcp = parm -> validate;
		if ((funcp NE NULL) AND (NOT (*funcp) (string))) {
			res = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
			break;
		}

		hookp = &PARAM_SLOT (param, parm->offset, char *);
		if (*hookp NE NULL) {
			free (*hookp);
		}
		*hookp = _gst_strdup (string);

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}

/*
 * Get the string for a 'string' parameter
 */

	int
gst_get_str_param (

gst_param_ptr		param,
int			whichparam,
int *			length,
char *			string
)
{
int			res;
const struct strparm *	parm;
char **			hookp;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		VALIDATE_STR (parm, param, whichparam);
		/* If we get here, parm has been set and is non-NULL. */

		hookp = &PARAM_SLOT (param, parm->offset, char *);

		if (length NE NULL) {
			*length = (*hookp NE NULL) ? strlen (*hookp) : -1;
		}

		if (string NE NULL) {
			*string = '\0';
			if (*hookp NE NULL) {
				strcpy (string, *hookp);
			}
		}

	} while (FALSE);

	GST_POSTLUDE

	return (res);
}
