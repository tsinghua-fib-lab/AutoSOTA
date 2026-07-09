#!/usr/bin/awk
#***********************************************************************
#
#	$Id: rename.awk,v 1.7 2022/11/19 13:45:54 warme Exp $
#
#	File:	rename.awk
#	Rev:	e-2
#	Date:	09/05/2016
#
#	Copyright (c) 2003, 2022 by Pawel Winter, Martin Zachariasen.
#	This work is licensed under a Creative Commons
#	Attribution-NonCommercial 4.0 International License.
#
#***********************************************************************
#
#	AWK script for generating #define's for rename.h.
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
# cat *.h | awk -f rename.awk > rename.h
#
/(extern.*[_a-zA-Z])+[	 ]*\(/ {
	for (field=3; field<=NF; ++field) {
		tmp = substr($field, 1, 1);
		if (tmp == "(") {
			printf ("#define %-37s _gst_%s\n", $(field-1), $(field-1));
		}
	}
}
