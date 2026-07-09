#!/usr/bin/awk
#***********************************************************************
#
#	$Id: bbox.awk,v 1.6 2022/11/19 13:45:49 warme Exp $
#
#	File:	bbox.awk
#	Rev:	e-2
#	Date:	09/05/2016
#
#	Copyright (c) 2003, 2022 by David M. Warme.  This work is
#	licensed under a Creative Commons Attribution-NonCommercial
#	4.0 International License.
#
#***********************************************************************
#
#	AWK script for generating bounding-box info.
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
/^%%EndComments/ { print "%%BoundingBox: 50 110 560 650" }
{ print }
