#!/usr/bin/awk
#***********************************************************************
#
#	$Id: verify.awk,v 1.6 2022/11/19 13:45:55 warme Exp $
#
#	File:	verify.awk
#	Rev:	e-2
#	Date:	09/05/2016
#
#	Copyright (c) 2002, 2022 by Pawel Winter & Martin Zachariasen.
#	This work is licensed under a Creative Commons
#	Attribution-NonCommercial 4.0 International License.
#
#***********************************************************************
#
# This script takes a look at all gst_-functions and verifies that they
# contain PRELUDE/POSTLUDE macros and that they only contain one return.
#
# You can run this script by writing:
# cat *.c | awk -f verify.awk | less
#
#***********************************************************************
#
#	Modification Log:
#
#	a-1:	02/12/2006	warme
#		: Added bannder and modlog.
#	e-1:	04/14/2015	warme
#		: Changes for 5.0 release.
#	e-2:	09/05/2016	warme
#		: Change notices for 5.1 release.
#
#***********************************************************************
#
/^gst(_[a-zA-Z]+)+ \((void\))*$/ { inFunction = 1; funcName = $1 }
/return/                         { if(inFunction == 2) { inFunction = 3 }
                                   if(inFunction == 1) { inFunction = 2 }
                                 }
/GST_PRELUDE/                    { if(prelude != 0)    { prelude = 10 }
				   if(inFunction > 0)  { prelude = 1 }
				 }
/GST_PRELUDE_NO_FP/              { noFP = 1 }
/GST_POSTLUDE/                   { if(prelude != 1)    { prelude = 10 }
				   if(prelude == 1)    { prelude = 2 }
				 }
/^}/ {

	if(inFunction != 0 && funcName != "gst_strdup")
	{
	   printf("gst-function: %s (%s)\n", funcName, noFP ? "No FP": "FP");
	   if(inFunction == 3) { printf(" Bug: More than 1 return!\n"); }

	   if(prelude == 0)    { printf(" Bug: Missing prelude!\n"); }
	   if(prelude == 1)    { printf(" Bug: Missing postlude!\n"); }
	   if(prelude == 10)   { printf(" Bug: Prelude/postlude mismatch!\n"); }

	   inFunction = 0;
	   prelude = 0;
	   noFP = 0;
	}

}
