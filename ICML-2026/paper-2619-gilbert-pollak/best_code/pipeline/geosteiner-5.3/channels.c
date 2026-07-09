/***********************************************************************

	$Id: channels.c,v 1.24 2022/11/19 13:45:50 warme Exp $

	File:	channels.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.
		: Fix -Wall issues.  Upgrade fatals.

************************************************************************/

#include "channels.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include "prepostlude.h"
#include "steiner.h"
#include <errno.h>
#include <stdarg.h>
#include <string.h>

/*
 * Global Routines
 */


static const gst_channel_options	default_channel_options = {
	0, 0, 0, 0
};

/*
 * Local Equates
 */

#define GST_CHNTYPE_FILE	1
#define GST_CHNTYPE_FUNCTION	2

/*
 * Local Types
 */

struct funcdest {
	gst_channel_func *	function;	/* func to call */
	void *			handle;		/* data to pass */
};

struct gst_destination {
	gst_channel_ptr		channel;	/* channel that owns us */
	gst_dest_ptr		prev;		/* previous dest */
	gst_dest_ptr		next;		/* next dest */

	int			type;		/* file or function */
	union {
		FILE *		filehandle;	/* open file output stream */
		struct funcdest	func;		/* function destination */
	} u;
};

/*
 * Local Routines
 */

static size_t	count_actual_bytes (gst_channel_ptr	chan,
				    const char *	text,
				    size_t		nbytes);
static void	edit_string (char *		buffer,
			     int		col,
			     gst_channel_ptr	chan,
			     const char *	text,
			     size_t		nbytes);
static int	write_dest (struct gst_destination *	dest,
			    const char *		buf,
			    size_t			nbytes);

/*
 * Create a channel with an optional set of options.
 */

	gst_channel_ptr
gst_create_channel (

const gst_channel_options *	opts,	/* IN - options */
int *				status	/* OUT - status */
)
{
int			i;
int			res;
struct gst_channel *	chan;

	GST_PRELUDE

	res = 0;

	chan = NEW (struct gst_channel);
	chan -> head = NULL;

	if (opts EQ NULL) {
		opts = &default_channel_options;
	}

	i = gst_channel_setopts (chan, opts);
	if (i NE 0) {
		res = i;
	}

	if (status NE NULL) {
		*status = res;
	}

	GST_POSTLUDE
	return chan;
}

/*
 * Freeing a channel and all its destinations.
 */

	int
gst_free_channel (

gst_channel_ptr		chan
)
{
gst_dest_ptr	dest;
gst_dest_ptr	next;

	GST_PRELUDE

	if (chan NE NULL) {
		dest = chan -> head;
		while (dest NE NULL) {
			next = dest -> next;
			/* Note: it is not our job to close streams, etc. */
			free (dest);
			dest = next;
		}

		free (chan);
	}

	GST_POSTLUDE
	return 0;
}

/*
 * Add a file destination to a channel.
 */

	gst_dest_ptr
gst_channel_add_file (

gst_channel_ptr		chan,
FILE *			fp,
int *			status
)
{
int				res;
struct gst_destination *	dest;
struct gst_destination *	p2;

	GST_PRELUDE

	res = 0;
	dest = NULL;

	do {		/* Used only for "break" */
		if (chan EQ NULL) {
			res = GST_ERR_INVALID_CHANNEL;
			break;
		}

		dest = NEW (struct gst_destination);

		memset (dest, 0, sizeof (*dest));

		p2 = chan -> head;

		dest -> channel		= chan;
		dest -> prev		= NULL;
		dest -> next		= p2;
		dest -> type		= GST_CHNTYPE_FILE;
		dest -> u.filehandle	= fp;

		if (p2 NE NULL) {
			p2 -> prev = dest;
		}

		chan -> head		= dest;
	} while (FALSE);

	if (status NE NULL) {
		*status = res;
	}

	GST_POSTLUDE
	return dest;
}

/*
 * Add a function as destination to a channel.
 */

	gst_dest_ptr
gst_channel_add_functor (

gst_channel_ptr			chan,
gst_channel_func *		func,
void *				handle,
int *				status
)
{
int				res;
struct gst_destination *	dest;
struct gst_destination *	p2;

	GST_PRELUDE

	res = 0;
	dest = NULL;

	do {		/* Used only for "break" */
		if (chan EQ NULL) {
			res = GST_ERR_INVALID_CHANNEL;
			break;
		}

		dest = NEW (struct gst_destination);

		memset (dest, 0, sizeof (*dest));

		p2 = chan -> head;

		dest -> channel		= chan;
		dest -> prev		= NULL;
		dest -> next		= p2;
		dest -> type		= GST_CHNTYPE_FUNCTION;
		dest -> u.func.function	= func;
		dest -> u.func.handle	= handle;

		if (p2 NE NULL) {
			p2 -> prev = dest;
		}

		chan -> head		= dest;
	} while (FALSE);

	if (status NE NULL) {
		*status = res;
	}

	GST_POSTLUDE
	return dest;
}

/*
 * Remove a destination from a channel.
 */

	int
gst_channel_rmdest (

gst_dest_ptr	dest
)
{
gst_dest_ptr	p1;
gst_dest_ptr	p2;

	GST_PRELUDE

	if (dest NE NULL) {
		p1 = dest -> prev;
		p2 = dest -> next;
		if (p1 NE NULL) {
			p1 -> next	= p2;
		}
		else {
			dest -> channel -> head = p2;
		}

		if (p2 NE NULL) {
			p2 -> prev	= p1;
		}

		free (dest);
	}

	GST_POSTLUDE
	return 0;
}

/*
 * Write a string to all destinations in a channel.
 */

	int
gst_channel_write (

gst_channel_ptr		chan,
const char *		text,
size_t			nbytes
)
{
int				i;
int				res;
int				column;
size_t				nactual;
gst_channel_options *		opts;
struct gst_destination *	dest;
char *				buffer;
const char *			wbuf;

	GST_PRELUDE

	res = 0;

	if (chan NE NULL) {
		opts = &(chan -> options);

		column = opts -> column;

		/* Count the actual number of bytes we will need to	*/
		/* write, including indentation, postscript comment	*/
		/* prefixes, etc.					*/
		nactual = count_actual_bytes (chan, text, nbytes);

		buffer = NULL;
		if (nactual > nbytes) {
			/* We need to edit the text being written.	*/
			/* Allocate a correctly sized buffer.		*/
			buffer = NEWA (nactual, char);

			/* Now edit the text, using the original	*/
			/* starting column.				*/
			edit_string (buffer, column, chan, text, nbytes);

			wbuf = buffer;
		}
		else {
			/* No editing.  Just write user's buffer. */
			wbuf = text;
		}

		for (dest = chan -> head; dest NE NULL; dest = dest -> next) {
			i = write_dest (dest, wbuf, nactual);
			if ((i NE 0) AND (res NE 0)) {
				res = i;
			}
		}

		if (buffer NE NULL) {
			free (buffer);
		}
	}

	GST_POSTLUDE
	return res;
}

/*
 * Count the actual number of bytes we will be writing.  This is normally
 * just nbytes, except that in certain cases (i.e., in column 0 and we
 * are either indenting or generating postscript comments) we need to
 * write extra stuff.
 */

	size_t
count_actual_bytes (

gst_channel_ptr		chan,
const char *		text,
size_t			nbytes
)
{
int			i;
int			col;
size_t			count;
char			c;
gst_channel_options *	opts;
const char *		p;
const char *		endp;

	opts = &(chan -> options);

	if ((opts -> indent <= 0) AND
	    (opts -> flags & GST_CHFLG_POSTSCRIPT) EQ 0) {
		/* No indent and no postscript.  Output is unedited. */
		return (nbytes);
	}

	/* Scan the string and count up the chars. */

	col = opts -> column;
	count = 0;

	p = text;
	endp = &text [nbytes];
	while (p < endp) {
		c = *p++;
		if ((col <= 0) AND (c NE '\n')) {
			/* Start of a new line.  Check for indent. */
			i = opts -> indent;
			while (i >= 8) {
				++count;	/* tab char */
				i -= 8;
			}
			count += i;		/* space chars */
			if ((opts -> flags & GST_CHFLG_POSTSCRIPT) NE 0) {
				count += 2;	/* "% " */
			}
		}
		++count;
		switch (c) {
		case '\n':	col = 0;			break;
		case '\f':	col = 0;			break;
		case '\t':	col = (col + 8) & ~0x07;	break;
		default:	++col;				break;
		}
	}

	opts -> column = col;

	return (count);
}

/*
 * Edit the given string (assuming the given starting column), inserting
 * all indentation and postscript comments required.
 */

	static
	void
edit_string (

char *		buffer,		/* IN/OUT - buf to put edited string into */
int		col,		/* IN - starting column number */
gst_channel_ptr	chan,		/* IN - channel we are editing for */
const char *	text,		/* IN - text string to edit */
size_t		nbytes		/* IN - number of bytes in text string */
)
{
int			i;
char			c;
gst_channel_options *	opts;
const char *		p;
const char *		endp;

	opts = &(chan -> options);

	/* Edit the string. */

	p = text;
	endp = &text [nbytes];
	while (p < endp) {
		c = *p++;
		if ((col <= 0) AND (c NE '\n')) {
			/* Start of a new line.  Check for indent. */
			i = opts -> indent;
			while (i >= 8) {
				*buffer++ = '\t';
				i -= 8;
			}
			while (i > 0) {
				*buffer++ = ' ';
				--i;
			}
			if ((opts -> flags & GST_CHFLG_POSTSCRIPT) NE 0) {
				*buffer++ = '%';
				*buffer++ = ' ';
			}
		}
		*buffer++ = c;
		switch (c) {
		case '\n':	col = 0;			break;
		case '\f':	col = 0;			break;
		case '\t':	col = (col + 8) & ~0x07;	break;
		default:	++col;				break;
		}
	}
}

/*
 * Write the given string to a single destination.
 */

	static
	int
write_dest (

struct gst_destination *	dest,	/* IN - destination to write to */
const char *			text,	/* IN - text buffer to write */
size_t				nbytes	/* IN - number of bytes to write */
)
{
int			res;
size_t			nwritten;
gst_channel_func *	funcp;

	res = 0;

	switch (dest -> type) {
	case GST_CHNTYPE_FILE:
		nwritten = fwrite (text,
				   sizeof (char),
				   nbytes,
				   dest -> u.filehandle);
		if (nwritten NE nbytes) {
			/* This is a write error! */
			res = errno;
		}
		break;

	case GST_CHNTYPE_FUNCTION:
		/* Note: user must pass write error codes back via handle. */
		funcp = dest -> u.func.function;
		(*funcp) (text, nbytes, dest -> u.func.handle);
		break;

	default:
		FATAL_ERROR;
	}

	return (res);
}

/*
 * 'printf' a string to all destinations in a channel.
 */

	int
gst_channel_printf (

gst_channel_ptr	chan,
const char *	format,
...
)
{
va_list		ap;
char		buf [1024]; /* How large should such a buffer be?!? */

	GST_PRELUDE

	va_start (ap, format);
	vsprintf (buf, format, ap);

	gst_channel_write (chan, buf, strlen (buf));

	va_end (ap);

	GST_POSTLUDE
	return 0;
}

/*
 * Get channel options.
 */

	int
gst_channel_getopts (

gst_channel_ptr		chan,
gst_channel_options *	options
)
{
int				res;
const gst_channel_options *	src;

	GST_PRELUDE

	res = 0;

	src = &default_channel_options;

	if (chan NE NULL) {
		src = &(chan -> options);
	}

	if (options EQ NULL) {
		res = GST_ERR_INVALID_CHANNEL_OPTIONS;
	}
	else {
		memcpy (options, src, sizeof (gst_channel_options));
	}

	GST_POSTLUDE
	return res;
}

/*
 * Set channel options.
 */

	int
gst_channel_setopts (

gst_channel_ptr			chan,
const gst_channel_options *	options
)
{
int		res;

	GST_PRELUDE

	res = 0;

	if (chan EQ NULL) {
		res = GST_ERR_INVALID_CHANNEL;
	}
	else if (options EQ NULL) {
		res = GST_ERR_INVALID_CHANNEL_OPTIONS;
	}
	else {
		memcpy (&(chan -> options),
			options,
			sizeof (gst_channel_options));
	}

	GST_POSTLUDE
	return res;
}
