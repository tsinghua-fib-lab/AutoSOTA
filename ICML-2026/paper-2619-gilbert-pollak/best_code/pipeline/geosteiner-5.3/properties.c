/***********************************************************************

	$Id: properties.c,v 1.25 2023/04/03 17:15:07 warme Exp $

	File:	properties.c
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	The property system.

************************************************************************

	Modification Log:

	a-1:	07/01/2002	benny
		: Created.
	b-1:	02/02/2014	warme
		: Removed an unnecessary include file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, upgrade fatals.
	e-4:	04/03/2023	warme
		: Added debugging routine to print a property list.

************************************************************************/

#include "fatal.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "prepostlude.h"
#include "solver.h"
#include "steiner.h"
#include <string.h>
#include "utils.h"

/*
 * Global Routines
 */

gst_proplist_ptr	gst_create_proplist (int *);
int			gst_free_proplist (gst_proplist_ptr);
int			gst_copy_proplist (gst_proplist_ptr, gst_proplist_ptr);
int			gst_delete_property (gst_proplist_ptr, int);
int			gst_get_property_type (gst_proplist_ptr, int, int *);

int			gst_get_dbl_property (gst_proplist_ptr, int, double *);
int			gst_get_int_property (gst_proplist_ptr, int, int *);
int			gst_get_str_property (gst_proplist_ptr,
					      int,
					      int *,
					      char *);
int			gst_get_properties (gst_proplist_ptr,
					    int *,
					    int *,
					    int *);
int			gst_set_dbl_property (gst_proplist_ptr, int, double);
int			gst_set_int_property (gst_proplist_ptr, int, int);
int			gst_set_str_property (gst_proplist_ptr, int, const char *);

/*
 * Local Types
 */

#define GST___PROPLIST_MAGIC 4242

#define VALID_PROPLIST(p) \
	(((p) NE NULL) AND ((p) -> magic EQ GST___PROPLIST_MAGIC))

typedef struct gst_property * gst_property_ptr;

struct gst_proplist {
	int magic;
	gst_property_ptr	first;
};

struct gst_property {
	int prop_id;	/* Property identity value: GST_PROP_... */
	int type;	/* int, double, string, (gst_proplist?) */
	struct gst_property *next;	/* The next property */

	union propval {
		int	ivalue;
		double	dvalue;
		char *	string;
	} u;
};

/*
 * Local Routines
 */

static void		copy_properties (gst_proplist_ptr, gst_proplist_ptr);
static void		free_properties (gst_proplist_ptr);
static int		get_property (gst_proplist_ptr,
				      int,
				      int,
				      struct gst_property **);
static int		make_property (gst_proplist_ptr,
				       int,
				       int,
				       struct gst_property **);

/*
 * Create an empty property list.
 */

	gst_proplist_ptr
gst_create_proplist (

int *		status		/* OUT - status value */
)
{
gst_proplist_ptr	tmp;

	GST_PRELUDE

	tmp = NEW (struct gst_proplist);
	memset (tmp, 0, sizeof (struct gst_proplist));
	tmp -> first = NULL;
	tmp -> magic = GST___PROPLIST_MAGIC;

	if (status NE NULL) {
		*status = 0;
	}

	GST_POSTLUDE
	return (tmp);
}

/*
 * Destroy a property list.
 */

	int
gst_free_proplist (

gst_proplist_ptr	plist	/* IN - property list to be freed */
)
{
int		res;

	GST_PRELUDE

	res = 0;

	if (plist NE NULL) {
		if (NOT VALID_PROPLIST (plist)) {
			res = GST_ERR_INVALID_PROPERTY_LIST;
		}
		else {
			free_properties (plist);
			free (plist);
		}
	}

	GST_POSTLUDE
	return res;
}

/*
 * Delete any entry in the given property list for the given property ID.
 */

	int
gst_delete_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id		/* IN - property ID to delete */
)
{
struct gst_property *	p;
struct gst_property **	hookp;

	if (NOT VALID_PROPLIST (plist)) {
		return GST_ERR_INVALID_PROPERTY_LIST;
	}

	hookp = &(plist -> first);
	for (;;) {
		p = *hookp;
		if (p EQ NULL) break;
		if (p -> prop_id EQ prop_id) {
			/* Delete this one... */
			*hookp = p -> next;
			if ((p -> type EQ GST_PROPTYPE_STRING) AND
			    (p -> u.string NE NULL)) {
				free (p -> u.string);
				p -> u.string = NULL;
			}
			free (p);
			return 0;
		}
		hookp = &(p -> next);
	}
	return GST_ERR_PROPERTY_NOT_FOUND;
}

/*
 * Return the type of the given property.
 */

	int
gst_get_property_type (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
int *			type		/* OUT - the type of the property */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = get_property (plist, prop_id, 0, &p);

	if (p NE NULL) {
		*type	= p -> type;
		res	= 0;
	}

	GST_POSTLUDE
	return res;
}

/*
 * Get the value of a 'dbl' property. An error is returned if the desired
 * property is not found in the list.
 */

	int
gst_get_dbl_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
double *		value		/* OUT - property value */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = get_property (plist, prop_id, GST_PROPTYPE_DOUBLE, &p);
	if ((res EQ 0) AND (value NE NULL)) {
		*value = p -> u.dvalue;
	}

	GST_POSTLUDE
	return res;
}

/*
 * Set the value of a 'dbl' property. If the value already exists in
 * the list of properties then the old value is overwritten. Otherwise
 * a new property is created and assigned the given value.
 */

	int
gst_set_dbl_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
double			value		/* IN - new property value */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = make_property (plist, prop_id, GST_PROPTYPE_DOUBLE, &p);
	if (p NE NULL) {
		p -> u.dvalue = value;
		res = 0;
	}

	GST_POSTLUDE
	return res;
}

/*
 * Get the value of a 'int' property. An error is returned if the desired
 * property is not found in the list.
 */

	int
gst_get_int_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
int *			value		/* OUT - property value */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = get_property (plist, prop_id, GST_PROPTYPE_INTEGER, &p);
	if ((res EQ 0) AND (value NE NULL)) {
		*value = p -> u.ivalue;
	}

	GST_POSTLUDE
	return res;
}

/*
 * Set the value of a 'int' property. If the value already exists in
 * the list of properties then the old value is overwritten. Otherwise
 * a new property is created and assigned the given value.
 */

	int
gst_set_int_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
int			value		/* IN - new property value */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = make_property (plist, prop_id, GST_PROPTYPE_INTEGER, &p);
	if (p NE NULL) {
		p -> u.ivalue = value;
		res = 0;
	}

	GST_POSTLUDE
	return res;
}

/*
 * Get the value of a 'str' property. An error is returned if the desired
 * property is not found in the list.
 */

	int
gst_get_str_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
int *			length,		/* OUT - string length */
char *			str		/* OUT - property value */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = get_property (plist, prop_id, GST_PROPTYPE_STRING, &p);
	if (res EQ 0) {
		if (length NE NULL) {
			*length = -1;
			if (p -> u.string NE NULL) {
				*length = strlen (p -> u.string);
			}
		}

		if (str NE NULL) {
			*str = '\0';
			if (p -> u.string NE NULL) {
				strcpy (str, p -> u.string);
			}
		}
	}

	GST_POSTLUDE
	return res;
}

/*
 * Set the value of a 'str' property. If the value already exists in
 * the list of properties then the old value is overwritten. Otherwise
 * a new property is created and assigned the given value.
 */

	int
gst_set_str_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
const char *		value		/* IN - new property value */
)
{
int			res;
struct gst_property *	p;

	GST_PRELUDE

	res = make_property (plist, prop_id, GST_PROPTYPE_STRING, &p);
	if (p NE NULL) {
		if (p -> u.string NE NULL) {
			free (p -> u.string);
			p -> u.string = NULL;
		}
		if (value NE NULL) {
			p -> u.string = _gst_strdup (value);
		}
		res = 0;
	}

	GST_POSTLUDE
	return res;
}

/*
 * Get the property IDs and the types of all properties currently
 * in the given property list.
 */

	int
gst_get_properties (

gst_proplist_ptr	plist,		/* IN - list of properties */
int *			count,		/* OUT - num properties in plist */
int *			propids,	/* OUT - property IDs */
int *			types		/* OUT - property types */
)
{
int			res;
int			n;
struct gst_property *	p;

	GST_PRELUDE

	if (NOT VALID_PROPLIST (plist)) {
		res = GST_ERR_INVALID_PROPERTY_LIST;
	}
	else {
		res = 0;
		if (count NE NULL) {
			n = 0;
			for (p = plist -> first; p NE NULL; p = p -> next) {
				++n;
			}
			*count = n;
		}
		if (propids NE NULL) {
			for (p = plist -> first; p NE NULL; p = p -> next) {
				*propids++ = p -> prop_id;
			}
		}
		if (types NE NULL) {
			for (p = plist -> first; p NE NULL; p = p -> next) {
				*types++ = p -> type;
			}
		}
	}

	GST_POSTLUDE
	return res;
}

/*
 * This routine makes a copy of a property list. If the source is NULL then
 * it will simply clear the destination list.
 */

	int
gst_copy_proplist (

gst_proplist_ptr	dest,
gst_proplist_ptr	src
)
{
int			res;

	GST_PRELUDE

	if ((NOT VALID_PROPLIST (dest)) OR
	    (NOT VALID_PROPLIST (src))) {
		res = GST_ERR_INVALID_PROPERTY_LIST;
	}
	else {
		if (dest NE src) {		/* avoid clobbering src! */
			free_properties (dest);
			copy_properties (dest, src);
		}
		res = 0;
	}

	GST_POSTLUDE
	return res;
}

/*
 * This routine copies a list of properties to an existing property list,
 * which we assume is initially empty.
 */

	static
	void
copy_properties (

gst_proplist_ptr	plist,
gst_proplist_ptr	src
)
{
gst_property_ptr	p;
gst_property_ptr *	hookp;
gst_property_ptr	newp;

	hookp = &(plist -> first);

	for (p = src -> first; p; p = p -> next) {
		newp = NEW (struct gst_property);
		memset (newp, 0, sizeof (*newp));

		newp -> prop_id = p -> prop_id;
		newp -> type	= p -> type;
		newp -> next	= NULL;

		switch (p -> type) {
		case GST_PROPTYPE_INTEGER:
			newp -> u.ivalue = p -> u.ivalue;
			break;

		case GST_PROPTYPE_DOUBLE:
			newp -> u.dvalue = p -> u.dvalue;
			break;

		case GST_PROPTYPE_STRING:
			if (p -> u.string EQ NULL) {
				newp -> u.string = NULL;
			}
			else {
				newp -> u.string = _gst_strdup (p -> u.string);
			}
			break;

		default:
			FATAL_ERROR;
		}

		*hookp = newp;
		hookp = &(newp -> next);
	}
}

/*
 * This routine frees a list of properties, but not the property list itself.
 */

	static
	void
free_properties (

gst_proplist_ptr	plist
)
{
gst_property_ptr	p;
gst_property_ptr	tmp;

	p = plist -> first;
	plist -> first = NULL;

	while (p NE NULL) {
		tmp = p -> next;
		if ((p -> type EQ GST_PROPTYPE_STRING) AND
		    (p -> u.string NE NULL)) {
			free (p -> u.string);
			p -> u.string = NULL;
		}
		free (p);
		p = tmp;
	}
}

/*
 * General routine that takes a list of properties and tries to find the
 * the desired property.
 */

	static
	int
get_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
int			type,		/* IN - property type */
struct gst_property **	pp		/* OUT - property pointer
						 if found/created */
)
{
struct gst_property *	p;

	if (NOT VALID_PROPLIST (plist)) {
		*pp = NULL;
		return GST_ERR_INVALID_PROPERTY_LIST;
	}

	/* Find property */
	for (p = plist -> first; p NE NULL; p = p -> next) {
		if (p -> prop_id EQ prop_id) {
			break;
		}
	}

	*pp = p;

	if (p EQ NULL) {
		return GST_ERR_PROPERTY_NOT_FOUND;
	}

	if (p -> type NE type) {
		return GST_ERR_PROPERTY_TYPE_MISMATCH;
	}

	return 0;
}

/*
 * General routine that takes a list of properties and tries to find the
 * the desired property.  If the property currently has a different type,
 * then the type is *forced* to the new type.  The property is created if
 * it does not currently exist.
 */

	static
	int
make_property (

gst_proplist_ptr	plist,		/* IN - list of properties */
int			prop_id,	/* IN - property id */
int			type,		/* IN - property type */
struct gst_property **	pp		/* OUT - property pointer
						 if found/created */
)
{
struct gst_property *	p;

	if (NOT VALID_PROPLIST (plist)) {
		*pp = NULL;
		return GST_ERR_INVALID_PROPERTY_LIST;
	}

	/* Find property */
	for (p = plist -> first; p NE NULL; p = p -> next) {
		if (p -> prop_id EQ prop_id) {
			break;
		}
	}

	if (p EQ NULL) {
		p = NEW (struct gst_property);
		memset (p, 0, sizeof(struct gst_property));
		p -> prop_id = prop_id;
		p -> type = type;
		p -> next = plist -> first;
		plist -> first = p;
	}

	*pp = p;

	if (p -> type NE type) {
		/* Changing property type -- assure consistent value. */
		switch (p -> type) {
		case GST_PROPTYPE_STRING:
			/* Changing from string to something else. */
			/* Free up string value, if present. */
			if (p -> u.string NE NULL) {
				free (p -> u.string);
				p -> u.string = NULL;
			}
			break;
		}

		p -> type = type;

		switch (type) {
		case GST_PROPTYPE_STRING:
			/* Changing to string from something else. */
			p -> u.string = NULL;
			break;
		}
	}

	return 0;
}

/*
 * Debug print the given property list.
 */

	void
_gst_debug_print_proplist (

gst_proplist_ptr	plist	/* IN - property list to print */
)
{
struct gst_property *	p;

	if (plist EQ NULL) {
		printf ("NULL property list pointer!\n");
		return;
	}
	if (NOT VALID_PROPLIST (plist)) {
		printf ("Not a valid property list pointer!\n");
		return;
	}

	for (p = plist -> first; p NE NULL; p = p -> next) {
		switch (p -> type) {
		case GST_PROPTYPE_INTEGER:
			printf ("%d\t%d\n", p -> prop_id, p -> u.ivalue);
			break;

		case GST_PROPTYPE_DOUBLE:
			printf ("%d\t%.17g\n", p -> prop_id, p -> u.dvalue);
			break;

		case GST_PROPTYPE_STRING:
			printf ("%d\t\"%s\"\n", p -> prop_id, p -> u.string);
			break;

		default:
			printf ("%d\t<Invalid type %d>\n",
				p -> prop_id, p -> type);
			break;
		}
	}
}
