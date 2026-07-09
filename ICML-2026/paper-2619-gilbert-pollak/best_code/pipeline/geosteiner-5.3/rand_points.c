/***********************************************************************

	$Id: rand_points.c,v 1.13 2023/04/03 17:10:49 warme Exp $

	File:	rand_points.c
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	This program generates random point sets.

************************************************************************

	Modification Log:

	a-1:	02/21/93	warme
		: Created.
	a-2:	02/03/96	warme
		: Embedded the code that my own system uses for
		:  "rand/srand" so that this program produces
		:  the same (bad) mapping between seeds and
		:  sequences on all systems.
		: Added a much better random number generator
		:  using 64-bit shift register with XOR feedback.
		:  Perhaps now the point sets will start "looking"
		:  more random...
	b-1:	02/28/2001	warme
		: Process command line arguments in a more standard way.
		: Add usage() routine for when crud is given.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Completely restructured this program to use separate
		:  modules for each pseudo-random number generator.
		: Added new generator based on AES-256.
		: Completely reworked command line switches:
		:  Added -b (binary mode).
		:  Added -g to select generator.
		:  Replaced -efo with -s seed_file.
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.
	e-4:	04/03/2023	warme
		: Fix inconsistent meaning and interpretation
		:  of return value from write_seed().

************************************************************************/

#include "rand_points.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/*
 * Global Routines
 */

int			main (int, char **);


/*
 * Local Routines
 */

static const struct PRNG_imp *
			choose_default_generator (void);
static const struct PRNG_imp *
			choose_generator_from_command_line (void);
static const struct PRNG_imp *
			choose_generator_from_seed_file (FILE ** fp_out);
static int32u		cv_number (char * num_string);
static void		decode_params (int argc, char ** argv);
static bool		seed_file_header_is_valid (FILE * fp);
static void		usage (void);
static void		write_seed_file_header (FILE *		  fp,
						struct PRNG_obj * obj);

/*
 * Local Variables
 */

static int		flag_digits;
static char *		flag_key;
static int		flag_generator;
static int		flag_num_points;
static enum Mode	flag_mode;
static int		flag_places;
static bool		flag_randomize;
static char *		flag_seed_file;
static char *		me;

/*
 * The magic header that we place at the beginning of all seed files.
 * This variable is '\0' terminated, but the nul byte does not appear
 * in the file.
 */

const char	seed_magic [] = "PRNG_STATE_V1\n\n";

/*
 * Table of PRNG implementations.
 */

static const struct PRNG_imp *	prng_imp_table [PRNG_NUM_IMPS];

/*
 * This routine generates random point sets.
 */

	int
main (

int		argc,
char **		argv
)
{
int			code;
bool			success;
const struct PRNG_imp *	imp;
const struct PRNG_imp *	default_imp;
const struct PRNG_imp *	cl_imp;
const struct PRNG_imp *	seed_imp;
struct PRNG_obj *	obj;
FILE *			fp;
struct PRNG_options	options;

	setbuf (stdout, NULL);

	prng_imp_table [PRNG_IMP_LEGACY]  = &rand_points_PRNG_legacy;
	prng_imp_table [PRNG_IMP_NEW]	  = &rand_points_PRNG_new;
	prng_imp_table [PRNG_IMP_AES_256] = &rand_points_PRNG_aes_256;

	decode_params (argc, argv);

	memset (&options, 0, sizeof (options));
	options.mode		= flag_mode;
	options.ndigits		= flag_digits;
	options.nplaces		= flag_places;
	options.randomize	= flag_randomize;
	if (flag_randomize) {
		/* Grab current time and use it to modify the seed. */
		options.cur_time	= time (NULL);
	}
	options.key		= flag_key;

	/* Choose which generator is the default. */

	default_imp	= choose_default_generator ();

	cl_imp		= choose_generator_from_command_line ();

	/* If a seed file has been specified, then it TELLS us which	*/
	/* generator to use.  (Seed file formats are different from one	*/
	/* generator to the next.)  Also returns the seed file input	*/
	/* stream, properly positioned to read in the			*/
	/* generator-specific seed state information.			*/
	fp = NULL;
	seed_imp	= choose_generator_from_seed_file (&fp);

	imp = NULL;
	if (seed_imp NE NULL) {
		if ((cl_imp NE NULL) AND (cl_imp NE seed_imp)) {
			fprintf (stderr,
				 "%s: Generator specified by seed file"
				 " conflicts with generator specified"
				 " on command line.\n",
				 me);
			if (fp NE NULL) {
				fclose (fp);
			}
			exit (1);
		}
		imp = seed_imp;
	}
	else {
		imp = cl_imp;
	}

	if (imp EQ NULL) {
		imp = default_imp;
	}

	FATAL_ERROR_IF (imp EQ NULL);

	/* Create an instance of the selected generator using the	*/
	/* given option settings.					*/

	obj = imp -> create (&options);

	if (obj EQ NULL) {
		/* Error message should have been printed by create(). */
		exit (1);
	}

	if (fp NE NULL) {
		success = obj -> read_seed (obj, fp);

		fclose (fp);

		if (NOT success) {
			fprintf (stderr,
				 "%s: Unable to read seed from `%s'.\n",
				 me, flag_seed_file);
			exit (1);
		}
	}

	obj -> gen_points (obj, stdout, flag_num_points);

	fflush (stdout);

	if (flag_seed_file) {
		fp = fopen (flag_seed_file, "w");
		if (fp EQ NULL) {
			fprintf (stderr,
				 "%s: Unable to write to seed file `%s'\n",
				 me, flag_seed_file);
			exit (1);
		}

		write_seed_file_header (fp, obj);

		code = obj -> write_seed (obj, fp);
		if (ferror (fp)) {
			code = 1;
		}
		fclose (fp);
		if (code NE 0) {
			fprintf (stderr,
				 "%s: Unable to write to seed file `%s'\n",
				 me, flag_seed_file);
			exit (1);
		}
	}

	obj -> destruct (obj);

	obj = NULL;

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

	--argc;
	me = *argv++;

	/* Allow each generator to choose its own defaults for	*/
	/* digits and places.					*/
	flag_mode	= MODE_DECIMAL;
	flag_digits	= -1;
	flag_places	= -1;		/* Make all digits be fractional. */

	flag_generator	= -1;

	flag_num_points	= 10;

	flag_key	= NULL;
	flag_randomize	= FALSE;
	flag_seed_file	= NULL;

#define	GET_FLAG_ARGUMENT(ap)				\
	do {						\
		if (*ap EQ '\0') {			\
			if (argc <= 0) {		\
				usage ();		\
			}				\
			ap = *argv++;			\
			--argc;				\
		}					\
	} while (FALSE)

	while (argc > 0) {
		ap = *argv++;
		--argc;
		if (*ap NE '-') {
			flag_num_points = cv_number (ap);
			if (((ap [0] < '0') AND (ap [0] > '9')) OR
			    (flag_num_points < 0)) {
				fprintf (stderr,
					 "%s: Invalid number of points `%s\n",
					 me, ap);
				usage ();
			}
			break;
		}
		++ap;
		while ((c = *ap++) NE '\0') {
			switch (c) {
			case 'b':
				flag_mode = MODE_BINARY;
				break;

			case 'd':
				GET_FLAG_ARGUMENT (ap);
				flag_digits = atoi (ap);
				if (flag_digits <= 0) {
					fprintf (stderr,
						 "%s: Invalid switch -d %d\n",
						 me, flag_digits);
					exit (1);
				}
				ap = "";
				break;

			case 'g':
				GET_FLAG_ARGUMENT (ap);
				flag_generator = atoi (ap);
				if (NOT PRNG_VALID_IMP_NUMBER (flag_generator)) {
					fprintf (stderr,
						 "%s:"
						 " Invalid generator"
						 " `%s'\n",
						 me, ap);
					usage ();
				}
				ap = "";
				break;

			case 'k':
				GET_FLAG_ARGUMENT (ap);
				flag_key = ap;
				ap = "";
				break;

			case 'p':
				GET_FLAG_ARGUMENT (ap);
				flag_places = atoi (ap);
				if (flag_places < 0) {
					fprintf (stderr,
						 "%s: Invalid switch -p %d\n",
						 me, flag_places);
					usage ();
				}
				ap = "";
				break;

			case 'r':
				flag_randomize = TRUE;
				break;

			case 's':
				GET_FLAG_ARGUMENT (ap);
				flag_seed_file = ap;
				ap = "";
				break;

			default:
				usage ();
				break;
			}
		}
	}

#undef GET_FLAG_ARGUMENT

	if (flag_mode EQ MODE_BINARY) {
		if (flag_digits NE -1) {
			fprintf (stderr, "%s: -b icompatible with -d\n", me);
			usage ();
		}
		if (flag_places NE -1) {
			fprintf (stderr, "%s: -b icompatible with -p\n", me);
			usage ();
		}
	}

	if (flag_places > flag_digits) {
		fprintf (stderr, "%s: -r exceed -d\n", me);
		usage ();
	}
}

/*
 * This routine prints out the proper usage and exits.
 */

static char *	arg_doc [] = {
	"",
	"\tGenerate N random points.  Default N is 10.",
	"",
	"\t-b\tBinary mode.  Generate coordinates that are uniformly",
	"\t\trandom IEEE doubles.",
	"\t-d N\tGenerate coordinates that are N digit decimal numbers.",
	"\t\tThe default N is 7 except for the legacy generator which",
	"\t\tdefaults to 4.  Format of these digits is controlled by",
	"\t\tthe -p switch.",
	"\t-g M\tChoose which pseudo-random number generator to use.",
	"\t\tValid choices are: 0 = legacy generator (poor),",
	"\t\t1 = `new' generator (better), 2 = generator that uses",
	"\t\tthe AES-256 cipher to produce randomness (excellent).",
	"\t\tDefaults to 2.  If the AES-256 generator is not",
	"\t\tavailable, the default is 1.  The default can be",
	"\t\toverridden with the RAND_POINTS_DEFAULT_GENERATOR",
	"\t\tenvironment variable.",
	"\t-k KEY\tUse the given KEY to alter the random sequence",
	"\t-m mode\tSet the given mode of operation: binary (b)",
	"\t\tor decimal (dN) where 1<=N<=15 is the number of decimal",
	"\t\tdigits to generate.  The default mode is \"d7\".",
	"\t-p M\tMake M of the digits be fractional (to right",
	"\t\tof the decimal point).  Only used in decimal mode.",
	"\t\tDefault is for all digits to be fractional.",
	"\t-r\tRandomize the generated point sequence by using",
	"\t\tthe current system time to generate the key/seed.",
	"\t-s path\tIf given path exists, read initial seed from",
	"\t\tthat file.  If given path does not exist, use other",
	"\t\targuments to determine the initial seed.  Regardless",
	"\t\tof whether the given path exists or not, that file",
	"\t\tis over-written with the final seed after all points",
	"\t\tare generated.",
	NULL
};

	static
	void
usage (void)

{
char **		pp;
char *		p;

	fprintf (stderr,
		 "\nUsage: %s [-br] [-d ndigits] [-g generator]"
		 " [-k key] [-p places] [-s seed_file] [num_points]\n",
		 me);

	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		fprintf (stderr, "%s\n", p);
	}
	exit (1);
}

/*
 * This routine converts the given string into a number.
 */

	static
	int32u
cv_number (

char *		num_string	/* IN - number to convert. */
)
{
char *		s;
int		c;
int32u		val;

	s = num_string;
	val = 0;
	while ((c = *s++) NE '\0') {
		if (NOT isdigit (c)) {
			(void) fprintf (stderr,
					"%s: `%s' is not a decimal number.\n",
					me, s);
			exit (1);
		}
		val = (val * 10) + (c - '0');
	}

	return (val);
}

/*
 * Decide which of the available generators should be the "default"
 * one to use (if user doesn't specify -g on command line).
 */

	static
	const struct PRNG_imp *
choose_default_generator (void)

{
int			i;
const char *		var;
char *			s;
const struct PRNG_imp *	imp;
const struct PRNG_imp *	default_imp;

	default_imp = NULL;

	/* Process the environment variable first. */
	var		= "RAND_POINTS_DEFAULT_GENERATOR";
	s		= getenv (var);
	do {		/* Used only for "break". */
		if (s EQ NULL) break;
		i = atoi (s);

		if (NOT PRNG_VALID_IMP_NUMBER (i)) {
			fprintf (stderr,
				 "%s: Warning:"
				 " invalid setting of `%s' environment"
				 " variable %s.\n",
				 me, s, var);
			break;
		}
		imp = prng_imp_table [i];
		if (PRNG_IMP_UNAVAILABLE (imp)) {
			fprintf (stderr,
				 "%s: Warning: environment variable %s"
				 " selects a generator `%s' that is not"
				 " available.\n", me, var, s);
		}
		else {
			/* Use default specified by environment variable. */
			default_imp = imp;
		}
	} while (FALSE);

	do {
		if (default_imp NE NULL) break;

		/* Use AES-256 (if present). */

		/* Prefer AES-256 (if present). */
		imp = prng_imp_table [PRNG_IMP_AES_256];
		if (PRNG_IMP_AVAILABLE (imp)) {
			default_imp = imp;
			break;
		}

		/* Fall back to `new', which should always be available. */
		imp = prng_imp_table [PRNG_IMP_NEW];
		FATAL_ERROR_IF (PRNG_IMP_UNAVAILABLE (imp));

		default_imp = imp;
	} while (FALSE);

	FATAL_ERROR_IF (default_imp EQ NULL);

	return (default_imp);
}

/*
 * Choose the generator specified on the command line.
 */

	static
	const struct PRNG_imp *
choose_generator_from_command_line (void)

{
int			i;
const struct PRNG_imp *	imp;

	imp = NULL;
	i = flag_generator;

	if (i < 0) {
		/* The -g flag was not specified. */
		return (NULL);
	}

	if (NOT (PRNG_VALID_IMP_NUMBER (i))) {
		fprintf (stderr,
			 "%s: Invalid generator specified (-g %d)\n",
			 me, i);
		exit (1);
	}
	else {
		imp = prng_imp_table [i];
		if (PRNG_IMP_UNAVAILABLE (imp)) {
			fprintf (stderr,
				 "%s: Specified generator (-g %d)"
				 " is not available.\n",
				 me, i);
			exit (1);
		}

	}

	return (imp);
}

/*
 * If a seed file has been specified, then it TELLS us which
 * generator to use.  (Seed file formats are different from one
 * generator to the next.)
 *
 * Also returns the seed file input stream, properly positioned to read
 * in the generator-specific seed state information.
 */

	static
	const struct PRNG_imp *
choose_generator_from_seed_file (

FILE **		fp_out		/* IN/OUT: seed file input stream */
)
{
int			i;
const struct PRNG_imp *	imp;
FILE *			fp;

	*fp_out = NULL;

	imp = NULL;
	fp = NULL;
	if (flag_seed_file NE NULL) {
		fp = fopen (flag_seed_file, "r");
		if (fp EQ NULL) {
			/* This is not an error!  We use the default	*/
			/* seed (possibly randomized or modified with	*/
			/* a key).  When we are done generating all of	*/
			/* the points, we will *create* the seed file	*/
			/* to remember the final seed state.		*/
		}
		else if (NOT seed_file_header_is_valid (fp)) {
			goto bad_seed_file;
		}
		else {
			if (fscanf (fp, " %d", &i) NE 1) {
				goto bad_seed_file;
			}
			if (NOT (PRNG_VALID_IMP_NUMBER (i))) {
				goto bad_seed_file;
			}
			imp = prng_imp_table [i];
			if (PRNG_IMP_UNAVAILABLE (imp)) {
				fprintf (stderr,
					 "%s: Generator specified by"
					 " seed file `%s' is not available.\n",
					 me, flag_seed_file);
				exit (1);
			}
		}
	}

	*fp_out = fp;

	return (imp);

bad_seed_file:
	/* Corrupt seed file is bad, so dispaly error and stop. */
	fprintf (stderr,
		 "%s: Seed file `%s' is invalid.\n",
		 me, flag_seed_file);
	exit (1);

	return (NULL);
}

/*
 * Return true if-and-only-if the given input stream contains a
 * valid seed file header.  If so, this header is read from the
 * stream.
 */

	static
	bool
seed_file_header_is_valid (

FILE *			fp		/* IN: input stream to read from */
)
{
size_t		nread, size;
char		buf [64];

	size = sizeof (seed_magic) - 1;
	nread = fread (buf, 1, size, fp);
	if (nread NE size) {
		return (FALSE);
	}

	if (memcmp (buf, seed_magic, size) NE 0) {
		return (FALSE);
	}

	/* The header is exactly what we expect. */

	return (TRUE);
}

/*
 * Write out the seed file header that indicates the given
 * type of generator.
 */

	static
	void
write_seed_file_header (

FILE *			fp,		/* IN/OUT: output stream to write to */
struct PRNG_obj *	obj		/* IN: generator to show in header */
)
{
	FATAL_ERROR_IF ((fp EQ NULL) OR (obj EQ NULL));

	fprintf (fp, "%s", seed_magic);

	fprintf (fp, "%d\n", obj -> type);
}
