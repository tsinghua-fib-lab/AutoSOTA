#!/bin/sh
#***********************************************************************
#
#	$Id: check_lib_syms.sh,v 1.6 2022/11/19 13:45:50 warme Exp $
#
#	File:	check_lib_syms.sh
#	Rev:	e-3
#	Date:	09/24/2016
#
#	Copyright (c) 2003, 2022 by David M. Warme.  This work is
#	licensed under a Creative Commons Attribution-NonCommercial
#	4.0 International License.
#
#***********************************************************************
#
#	A shell script to display all symbols in the library that
#	violate the GeoSteiner naming conventions.
#
#***********************************************************************
#
#	Modification Log:
#
#	a-1:	08/25/2003	warme
#		: Created.
#	e-1:	04/14/2015	warme
#		: Changes for 5.0 release.
#	e-2:	09/05/2016	warme
#		: Change notices for 5.1 release.
#	e-3:	09/24/2016	warme
#		: Modified for 64-bit architecture.
#
#***********************************************************************

TMPDIR=tmpdir.$$

(rm -rf $TMPDIR
 mkdir $TMPDIR
 cd $TMPDIR
 ar -x ../libgeosteiner.a

 for i in *.o
 do
   nm -p $i | \
   grep -v ':$' | \
   grep '^................ [CDT] ' | \
   egrep -v '^................ [CDT] (gst_|_gst_)' | \
   sed -e 's/^................ [CDT] //' | \
   sort -u | \
   sed -e "s/^/${i}:	/"
 done

 cd ..
 rm -rf $TMPDIR
)
