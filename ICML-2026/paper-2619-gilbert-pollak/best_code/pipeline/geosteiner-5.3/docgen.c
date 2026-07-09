/***********************************************************************

	$Id: docgen.c,v 1.34 2022/11/19 13:45:51 warme Exp $

	File:	docgen.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Output header information or manual information based on formatted
	input (functions.in).

************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
		: Add a banner section to the front of the input that is
		:  ignored in every mode of operation.  This lets us
		:  put legal notices / mod log in the functions.in file.
	e-2:	09/05/2016	warme
		: Added compile-time checking of printf args.
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.
	e-4:	11/18/2022	warme
		: Fixed -Wall issue.

************************************************************************/

#include "config.h"
#include "errordefs.h"
#include "gsttypes.h"
#include "logic.h"
#include "parmdefs.h"
#include "propdefs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Local Routines
 */

static void		decode_params (int, char **);
static void		strip_latex (char *);
static void		usage (void);
static char *		texify (const char *);

/*
 * Local Variables
 */

static char *		me;
static bool		Print_Header = FALSE;
static bool		Print_Function_Descriptions = FALSE;
static bool		Print_Error_Values = FALSE;
static bool		Print_Hypergraph_Properties = FALSE;
static bool		Print_Solver_Properties = FALSE;

#define INTROW(sym,val,var,min,max,dflt) {#sym, val},
#define DBLROW(sym,val,var,min,max,dflt) {#sym, val},
#define STRROW(sym,val,var,func,dflt) {#sym, val},
#define CHNROW(sym,val,var,func) {#sym, val},
#define ERRROW(sym,val) {#sym, val},

#define MAX_LINE_LENGTH 2048

struct keywords {
	char *	name;
	int	type;
	char *	header;
	char *	footer;
	char *	prefix;
	int	length;
};

enum {	SECTION = 0,
	FUNCNAME,
	DESCRIPTION,
	FUNCTION,
	ARGUMENTS,
	RETURNVALUE,
	EXAMPLE,
	COMMENT,
	HEADERINFO,
	BANNER,
	NUM_OF_KEYS
};

#define VERBATIM	0
#define SKIPNEWLINES	1
#define SINGLEWORD	2
#define SINGLELINE	3
#define IGNORE		4
#define LATEX		5

struct keywords document_kw[] = {
	{ "SECTION",		SINGLEWORD,	"\\clearpage\\subsection{", "}\n", "" },
	{ "FUNCNAME",		SINGLEWORD,	"\\clearpage", "\n\\hrule\n\\vskip 0.25in\n", "" },
	{ "DESCRIPTION",	LATEX,		"", "\n", "" },
	{ "FUNCTION",		SKIPNEWLINES,	"\\begin{verbatim}\n", "\n\\end{verbatim}\n", "" },
	{ "ARGUMENTS",		VERBATIM,	"\n\\begin{tabular}{ll}\n~\\hspace*{3cm} & \\hspace*{8cm}\\\\ \\hline\n",
						"\\end{tabular}\n\n", "" },
	{ "RETURNVALUE",	LATEX,		"", "\n", "" },
	{ "EXAMPLE",		VERBATIM,	"\\bigskip{}Example:\n{\\footnotesize\n\\begin{verbatim}\n",
						"\\end{verbatim}\n}\n", "" },
	{ "COMMENT",		SINGLELINE,	"\\comment{", "}\n", "" },
	{ "HEADERINFO",		IGNORE,		"", "", "" },
	{ "BANNER",		IGNORE,		"", "", "" }
};

struct keywords header_kw[] = {
	{ "SECTION",		SINGLEWORD,	"/****************************************************************/\n\n/*\n * ",
						"\n * \n", ""},
	{ "FUNCNAME",		SINGLEWORD,	"/****************************************/\n\n/*\n * ",
						"\n * \n", ""},
	{ "DESCRIPTION",	LATEX,		"", " */\n\n", " * "},
	{ "FUNCTION",		SKIPNEWLINES,	"", "\n", ""},
	{ "ARGUMENTS",		IGNORE,		"", "", ""},

	{ "RETURNVALUE",	LATEX,		"/*\n", " */\n\n",  " * "},
	{ "EXAMPLE",		IGNORE,		"", "", ""},

	{ "COMMENT",		IGNORE,		"", "\n", ""},
	{ "HEADERINFO",		VERBATIM,	"", "\n", ""},
	{ "BANNER",		IGNORE,		"", "", ""}
};

static const char *tex_header = "";

static const char *tex_footer = "";

static const char *h_header =
	"/***********************************************************************\n"
	"\n"
	" GeoSteiner " GEOLIB_VERSION_STRING " header file\n"
	"\n"
	" Copyright (c) 2004, 2022 by David M. Warme, Pawel Winter & Martin Zachariasen.\n"
	" All rights reserved.\n"
	"\n"
	"************************************************************************/\n"
	"\n"
	"#ifndef GEOSTEINER_H\n"
	"#define GEOSTEINER_H\n"
	"\n"
	"#include <stdio.h>\n"
	"#include <stddef.h>\n"
	"\n"
	"#ifdef __cplusplus\n"
	"extern \"C\" {\n"
	"#endif\n"
	"\n"
	"#ifdef __GNUC__\n"
	" #define _GST_PRINTF_ARGS(a,b) __attribute__ ((__format__ (__printf__,a,b)))\n"
	"#else\n"
	" #define _GST_PRINTF_ARGS(a,b)\n"
	"#endif\n"
	"\n"
	;

static const char *h_footer =
	"\n"
	"#ifdef __cplusplus\n"
	"}\n"
	"#endif\n"
	"\n"
	"#endif\n";

struct parmdef {
	const char *	symbol;
	int		value;
};

const struct parmdef	parmtable [] = {
	INTPARMS(INTROW)
	DBLPARMS(DBLROW)
	STRPARMS(STRROW)
	CHNPARMS(CHNROW)
	{NULL, 0},
};

const struct parmdef	errortable [] = {
	ERRORVALS(ERRROW)
	{NULL, 0},
};

const struct parmdef	hgtable [] = {
	HG_PROPS(ERRROW)
	{NULL, 0},
};

const struct parmdef	solvertable [] = {
	SOLVER_PROPS(ERRROW)
	{NULL, 0},
};

	int
main (

int		argc,	/* IN - argument count */
char **		argv	/* IN - arguments */
)
{
int		i;
char *		footer;
char		buffer[MAX_LINE_LENGTH];
char *		tmp;
bool		newline;
struct keywords *kw;
const struct parmdef *	p;

	decode_params (argc, argv);

	if (Print_Error_Values) {
		printf ("\\begin{tabular}{|l|l|}\n");
		printf ("\\hline\n");
		printf ("Error Code & Value \\\\\n");
		printf ("\\hline\n");
		for (p = &errortable [0]; p -> symbol NE NULL; p++) {
#if 0
			printf ("GST\\_ERR\\_%s \\\\\n", texify (p -> symbol));
#endif
			printf ("GST\\_ERR\\_%s & %d \\\\\n", texify (p -> symbol), p -> value);
		}
		printf ("\\hline\n");
		printf ("\\end{tabular}\n");
		exit (0);
	}

	if (Print_Hypergraph_Properties) {
		printf ("\\begin{tabular}{|l|l|}\n");
		printf ("\\hline\n");
		printf ("Property & Value \\\\\n");
		printf ("\\hline\n");
		for (p = &hgtable [0]; p -> symbol NE NULL; p++) {
#if 0
			printf ("GST\\_PROP\\_HG\\_%s \\\\\n", texify (p -> symbol));
#endif
			printf ("GST\\_PROP\\_HG\\_%s & %d \\\\\n", texify (p -> symbol), p -> value);
		}
		printf ("\\hline\n");
		printf ("\\end{tabular}\n");
		exit (0);
	}

	if (Print_Solver_Properties) {
		printf ("\\begin{tabular}{|l|l|}\n");
		printf ("\\hline\n");
		printf ("Property & Value \\\\\n");
		printf ("\\hline\n");
		for (p = &solvertable [0]; p -> symbol NE NULL; p++) {
#if 0
			printf ("GST\\_PROP\\_SOLVER\\_%s \\\\\n", texify (p -> symbol));
#endif
			printf ("GST\\_PROP\\_SOLVER\\_%s & %d \\\\\n", texify (p -> symbol), p -> value);
		}
		printf ("\\hline\n");
		printf ("\\end{tabular}\n");
		exit (0);
	}

	if (Print_Function_Descriptions) {
		printf ("%s", tex_header);
		kw = document_kw;
	}

	if (Print_Header) {
		printf ("%s", h_header);

		/* Write the list of error values in the header file */
		printf ("/* Hypergraph properties */\n\n");
		for (p = &hgtable [0]; p -> symbol NE NULL; p++) {
			printf ("#define GST_PROP_HG_%-38s%d\n",
				p -> symbol, p -> value);
		}

		/* Write the list of error values in the header file */
		printf ("\n/* Solver properties */\n\n");
		for (p = &solvertable [0]; p -> symbol NE NULL; p++) {
			printf ("#define GST_PROP_SOLVER_%-34s%d\n",
				p -> symbol, p -> value);
		}

		/* Write the list of error values in the header file */
		printf ("\n/* Error codes */\n\n");
		for (p = &errortable [0]; p -> symbol NE NULL; p++) {
			printf ("#define GST_ERR_%-42s%d\n",
				p -> symbol, p -> value);
		}

		/* Write the list of parameters in the header file */
		printf ("\n/* Parameters */\n\n");
		for (p = &parmtable [0]; p -> symbol NE NULL; p++) {
			printf ("#define GST_PARAM_%-40s%d\n",
				p -> symbol, p -> value);
		}

		kw = header_kw;
	}

	for (i=0; i < NUM_OF_KEYS; i++) {
		kw[i].length = strlen (kw[i].name);
	}

	i = BANNER;
	newline = FALSE;
	footer = "";
	buffer[0] = '\0';
	while (buffer[0] OR fgets(buffer, MAX_LINE_LENGTH, stdin)) {
		if (*buffer EQ '@') {
			newline = FALSE;
			i = 0;
			while (i < NUM_OF_KEYS) {
				if (!strncmp (kw[i].name, &buffer[1], kw[i].length))
					break;
				i++;
			}

			if (i >= NUM_OF_KEYS) {
				fprintf (stderr, "docgen error: Unknown keyword; %s", buffer);
				abort ();
			}

			printf("%s", footer);
			printf("%s", kw[i].header);
			footer = kw[i].footer;

			if (i EQ ARGUMENTS) {
				char *arg_footer = "";
				while (fgets(buffer, MAX_LINE_LENGTH, stdin)) {
					if (buffer[0] EQ '%') {}
					else if (buffer[0] EQ '@') {
						if (kw[i].type != IGNORE) {
							printf("%s", arg_footer);
						}
						if(buffer[1] EQ 'A') {
							char var[256];
							sscanf (buffer, "@A %s\n", var);
							if (kw[i].type != IGNORE) {
								printf("\\code{%s} &\n\\adescr{", var);
							}
							arg_footer = "}\\\\\n\\hline\n";
						}
						else
							break;
					}
					else {
						buffer[strlen(buffer)-1] = ' ';
						if (kw[i].type != IGNORE) {
							printf("%s", buffer);
						}
					}
				}

			}
			else
				*buffer = '\0';
		}
		else {
			if (kw[i].type EQ IGNORE) {
				/* Ignore it */
			}
			else if (    (buffer[0] EQ '\n')
			    AND (kw[i].type EQ SKIPNEWLINES)) {
				/* Ignoring newlines */
			}
			else if (buffer[0] EQ '%') {
				/* Always ignore comments */
			}
			else {
				if (kw[i].type EQ SINGLEWORD) {
					buffer[strlen(buffer)-1] = '\0';
				}
				if (kw[i].type EQ SINGLELINE) {
					buffer[strlen(buffer)-1] = ' ';
				}
				if (buffer[0] EQ '\n') {
					newline = TRUE;
				}
				else {
					if (newline EQ TRUE) {
						printf("%s\n", kw[i].prefix);
						newline = FALSE;
					}
					if (Print_Header) {
						if (   kw[i].type EQ SINGLELINE
						    OR kw[i].type EQ LATEX) {
							strip_latex (buffer);
						}
					}
					if (   (i EQ FUNCNAME)/* Special case */
					    AND Print_Function_Descriptions
					    AND (*buffer NE '\0')) {
						tmp = texify (buffer);
						printf("\\func{%s}\n"
						       "\\label{%s}\n"
						       "\\index{%s}\n",
						       tmp, buffer, tmp);
					}
					else {
						printf("%s%s", kw[i].prefix, buffer);
					}
				}
			}
			*buffer = '\0';
		}
	}
	if (Print_Function_Descriptions) {
		printf("%s\n", footer);
		printf("%s", tex_footer);
	}
	if (Print_Header) {
		printf("%s", h_footer);
	}

	exit (0);
}

/*
 * This routine decodes the various command-line arguments.
 */

	static
	void
decode_params (

int		argc,
char **		argv
)
{
char *		ap;
char		c;
int		flags;

	flags = 0;
	--argc;
	me = *argv++;
	while (argc > 0) {
		ap = *argv++;
		if (*ap NE '-') {
			usage ();
		}
		++ap;
		while ((c = *ap++) NE '\0') {
			switch (c) {
			case 'e':
				Print_Error_Values = TRUE;
				break;

			case 'f':
				Print_Function_Descriptions = TRUE;
				break;

			case 'g':
				Print_Hypergraph_Properties = TRUE;
				break;

			case 'h':
				Print_Header = TRUE;
				break;

			case 's':
				Print_Solver_Properties = TRUE;
				break;

			default:
				usage ();
				break;
			}
			flags++;
		}
		--argc;
	}

	if (flags NE 1) {
		usage ();
	}
}


/*
 * This routine prints out the proper usage and exits.
 */

static char *	arg_doc [] = {
	"",
	"\t-e\tPrint the error values.",
	"\t-f\tPrint the function descriptions.",
	"\t-g\tPrint the hypergraph properties.",
	"\t-h\tPrint the header file.",
	"\t-s\tPrint the solver properties.",
	"",
	NULL
};

	static
	void
usage (void)

{
char **		pp;
char *		p;

	(void) fprintf (stderr,
			"\nUsage: %s [-efh]"
			" <functions-file\n",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}

/*
 * This routine strips latex-code from a given string.
 * It can strip \keyword and {}.
 * The routine below does currently not allow '{', '}' and '\' to be
 * escaped in any way.
 */

enum { NORMAL = 0, STRIP};
static int state = NORMAL;

	static
	void
strip_latex (

char *	buffer
)
{
char *	output;

	output = buffer;
	while (*buffer NE '\0') {
		switch (*buffer) {
		case '{':
		case '}':
			break;
		case '\\':
			state = STRIP;
			break;
		case ' ':
			if (state EQ STRIP) {
				state = NORMAL;
			}
			else {
				*output++ = *buffer;
			}
			break;
		default:
			if (	((*buffer < 'a') OR (*buffer > 'z'))
			    AND ((*buffer < 'A') OR (*buffer > 'B'))) {
				state = NORMAL;
			}

			if (state NE STRIP) {
				*output++ = *buffer;
			}
			break;
		}

		buffer++;
	}
	*output = '\0';
}

/*
 * This routine escapes certain latex sensitive characters.
 */
	static
	char *
texify (

const char *		input		/* IN - simple string */
)
{
char *		buffer;
char *		output;

	buffer = output = malloc (2*strlen(input) + 1);
	while (*input NE '\0') {
		switch (*input) {
		case ' ':	/* Ignore spaces */
			break;
		case '\\':	/* Don't touch any escaped characters */
			*output++ = *input++;
			*output++ = *input;
			break;
		case '_':	/* Escape this character */
			*output++ = '\\';
			/* break; */
		default:
			*output++ = *input;
			break;
		}

		input++;
	}
	*output = '\0';

	return buffer;
}
