/***********************************************************************

	$Id: triangle_rename.h,v 1.6 2022/11/19 13:45:55 warme Exp $

	File:	triangle_rename.h
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Defines to keep names from the "triangle" package from
	polluting the library namespace.

************************************************************************

	Modification Log:

	a-1:	08/25/2003	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

************************************************************************/

#ifndef TRIANGLE_RENAME_H
#define	TRIANGLE_RENAME_H

#define alternateaxes			_gst_alternateaxes
#define areaboundindex			_gst_areaboundindex
#define badsegments			_gst_badsegments
#define badtriangles			_gst_badtriangles
#define carveholes			_gst_carveholes
#define ccwerrboundA			_gst_ccwerrboundA
#define ccwerrboundB			_gst_ccwerrboundB
#define ccwerrboundC			_gst_ccwerrboundC
#define checksegments			_gst_checksegments
#define circletopcount			_gst_circletopcount
#define circumcentercount		_gst_circumcentercount
#define constrainededge			_gst_constrainededge
#define convex				_gst_convex
#define counterclockcount		_gst_counterclockcount
#define counterclockwise		_gst_counterclockwise
#define counterclockwiseadapt		_gst_counterclockwiseadapt
#define delaunay			_gst_delaunay
#define delaunayfixup			_gst_delaunayfixup
#define divconqdelaunay			_gst_divconqdelaunay
#define divconqrecurse			_gst_divconqrecurse
#define docheck				_gst_docheck
#define dummyinit			_gst_dummyinit
#define dummysh				_gst_dummysh
#define dummyshbase			_gst_dummyshbase
#define dummytri			_gst_dummytri
#define dummytribase			_gst_dummytribase
#define dwyer				_gst_dwyer
#define edges				_gst_edges
#define edgesout			_gst_edgesout
#define eextras				_gst_eextras
#define elemattribindex			_gst_elemattribindex
#define epsilon				_gst_epsilon
#define estimate			_gst_estimate
#define exactinit			_gst_exactinit
#define fast_expansion_sum_zeroelim	_gst_fast_expansion_sum_zeroelim
#define findcircumcenter		_gst_findcircumcenter
#define finddirection			_gst_finddirection
#define firstnumber			_gst_firstnumber
#define fixedarea			_gst_fixedarea
#define flip				_gst_flip
#define formskeleton			_gst_formskeleton
#define geomview			_gst_geomview
#define getpoint			_gst_getpoint
#define goodangle			_gst_goodangle
#define highorder			_gst_highorder
#define highorderindex			_gst_highorderindex
#define holes				_gst_holes
#define hullsize			_gst_hullsize
#define hyperbolacount			_gst_hyperbolacount
#define iccerrboundA			_gst_iccerrboundA
#define iccerrboundB			_gst_iccerrboundB
#define iccerrboundC			_gst_iccerrboundC
#define incircle			_gst_incircle
#define incircleadapt			_gst_incircleadapt
#define incirclecount			_gst_incirclecount
#define incremental			_gst_incremental
#define inelements			_gst_inelements
#define infecthull			_gst_infecthull
#define infpoint1			_gst_infpoint1
#define infpoint2			_gst_infpoint2
#define infpoint3			_gst_infpoint3
#define initializepointpool		_gst_initializepointpool
#define initializetrisegpools		_gst_initializetrisegpools
#define inpoints			_gst_inpoints
#define insegments			_gst_insegments
#define insertsegment			_gst_insertsegment
#define insertshelle			_gst_insertshelle
#define insertsite			_gst_insertsite
#define internalerror			_gst_internalerror
#define locate				_gst_locate
#define makepointmap			_gst_makepointmap
#define makeshelle			_gst_makeshelle
#define maketriangle			_gst_maketriangle
#define markhull			_gst_markhull
#define maxarea				_gst_maxarea
#define mergehulls			_gst_mergehulls
#define mesh_dim			_gst_mesh_dim
#define minangle			_gst_minangle
#define minus1mod3			_gst_minus1mod3
#define neighbors			_gst_neighbors
#define nextras				_gst_nextras
#define nobisect			_gst_nobisect
#define nobound				_gst_nobound
#define noelewritten			_gst_noelewritten
#define noexact				_gst_noexact
#define noholes				_gst_noholes
#define noiterationnum			_gst_noiterationnum
#define nonodewritten			_gst_nonodewritten
#define nopolywritten			_gst_nopolywritten
#define numbernodes			_gst_numbernodes
#define order				_gst_order
#define parsecommandline		_gst_parsecommandline
#define plague				_gst_plague
#define plus1mod3			_gst_plus1mod3
#define point2triindex			_gst_point2triindex
#define pointdealloc			_gst_pointdealloc
#define pointmarkindex			_gst_pointmarkindex
#define pointmedian			_gst_pointmedian
#define points				_gst_points
#define pointsort			_gst_pointsort
#define pointtraverse			_gst_pointtraverse
#define poly				_gst_poly
#define poolalloc			_gst_poolalloc
#define pooldealloc			_gst_pooldealloc
#define pooldeinit			_gst_pooldeinit
#define poolinit			_gst_poolinit
#define poolrestart			_gst_poolrestart
#define preciselocate			_gst_preciselocate
#define printshelle			_gst_printshelle
#define printtriangle			_gst_printtriangle
#define quality				_gst_quality
#define quality_statistics		_gst_quality_statistics
#define queuefront			_gst_queuefront
#define queuetail			_gst_queuetail
#define quiet				_gst_quiet
#define randomnation			_gst_randomnation
#define randomseed			_gst_randomseed
#define readnodefile			_gst_readnodefile
#define recenttri			_gst_recenttri
#define refine				_gst_refine
#define regionattrib			_gst_regionattrib
#define regionplague			_gst_regionplague
#define regions				_gst_regions
#define removeghosts			_gst_removeghosts
#define resulterrbound			_gst_resulterrbound
#define samples				_gst_samples
#define scale_expansion_zeroelim	_gst_scale_expansion_zeroelim
#define scoutsegment			_gst_scoutsegment
#define segmentintersection		_gst_segmentintersection
#define shelledealloc			_gst_shelledealloc
#define shelles				_gst_shelles
#define shelletraverse			_gst_shelletraverse
#define shwords				_gst_shwords
#define splaynodes			_gst_splaynodes
#define splitseg			_gst_splitseg
#define splitter			_gst_splitter
#define statistics			_gst_statistics
#define steiner				_gst_steiner
#define steinerleft			_gst_steinerleft
#define sweepline			_gst_sweepline
#define transfernodes			_gst_transfernodes
#define traversalinit			_gst_traversalinit
#define traverse			_gst_traverse
#define triangledealloc			_gst_triangledealloc
#define triangledeinit			_gst_triangledeinit
#define triangleinit			_gst_triangleinit
#define triangles			_gst_triangles
#define triangletraverse		_gst_triangletraverse
#define triangulatepolygon		_gst_triangulatepolygon
#define triwords			_gst_triwords
#define useshelles			_gst_useshelles
#define vararea				_gst_vararea
#define verbose				_gst_verbose
#define viri				_gst_viri
#define voronoi				_gst_voronoi
#define writeedges			_gst_writeedges
#define writeelements			_gst_writeelements
#define writeneighbors			_gst_writeneighbors
#define writenodes			_gst_writenodes
#define writepoly			_gst_writepoly
#define writevoronoi			_gst_writevoronoi
#define xmax				_gst_xmax
#define xmin				_gst_xmin
#define xminextreme			_gst_xminextreme
#define ymax				_gst_ymax
#define ymin				_gst_ymin

#endif
