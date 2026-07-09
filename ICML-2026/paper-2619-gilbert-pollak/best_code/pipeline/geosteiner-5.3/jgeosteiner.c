/***********************************************************************

	$Id: jgeosteiner.c,v 1.9 2022/11/19 13:45:52 warme Exp $

	File:	jgeosteiner.c
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	A JNI interface for GeoSteiner.

************************************************************************

	Modification Log:

	a-1:	02/12/2006	warme
		: Added banner and modlog.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

************************************************************************/

#ifndef JNICALL
#define JNICALL
#endif

#ifndef JNIEXPORT
#define JNIEXPORT
#endif

#include "jgeosteiner.h"
#include "geosteiner.h"
#include <jni.h>

void Java_jgeosteiner_open (JNIEnv *env, jobject obj)
{
	gst_open_geosteiner();
}

void Java_jgeosteiner_close (JNIEnv *env, jobject obj)
{
	gst_close_geosteiner();
}

jdouble Java_jgeosteiner_SMT (

JNIEnv *		env,
jobject			obj,
jint			lambda,
jint			max_fst_size,
jdoubleArray		jterms,
jintArray		jnsteins,
jdoubleArray		jstps,
jintArray		jnedges,
jintArray		jedges
)
{
int		i;
double		smtlength = 0;
gst_metric_ptr	metric;
gst_param_ptr	params;

	int nterms = ((*env)->GetArrayLength(env, jterms)) / 2;
	jdouble *terms	= (*env)->GetDoubleArrayElements(env, jterms, 0);

	jint *nsteins 	= (*env)->GetIntArrayElements(env, jnsteins, 0);
	jdouble *stps 	= (*env)->GetDoubleArrayElements(env, jstps, 0);

	jint *nedges 	= (*env)->GetIntArrayElements(env, jnedges, 0);
	jint *edges 	= (*env)->GetIntArrayElements(env, jedges, 0);

	if (lambda < 0) {
		fprintf (stderr, "Java_jgeosteiner_SMT: Lambda must be greater"
				 "than or equal to 0.\n");
		abort ();
	}

	/* Use GeoSteiner to find a solution */
	switch(lambda)
	{
	case 0:		/* The Euclidean metric */
		metric = gst_create_metric (GST_METRIC_L, 2, NULL);
		break;
	case 1:		/* The rectilinear metric */
		metric = gst_create_metric (GST_METRIC_L, 1, NULL);
		break;
	default:	/* The uniform metric (lambda >= 2) */
		metric = gst_create_metric (GST_METRIC_UNIFORM, lambda, NULL);
		break;
	}

	gst_param_ptr params = gst_create_param(NULL);

	/* This enables the visually important corner points of lambda trees */
	gst_set_int_param (params, GST_PARAM_INCLUDE_CORNERS, 1);

	/* And optionally we only require FSTs of some maximal size */
	if (max_fst_size > 0) {
		gst_set_int_param (params, GST_PARAM_MAX_FST_SIZE, max_fst_size);
	}

	gst_smt(nterms, terms, &smtlength, &nsteins[0], stps,
		&nedges[0], edges, NULL, metric, params);

	(*env)->ReleaseIntArrayElements(env, jnsteins, nsteins, 0);
	(*env)->ReleaseDoubleArrayElements(env, jstps, stps, 0);

	(*env)->ReleaseIntArrayElements(env, jnedges, nedges, 0);
	(*env)->ReleaseIntArrayElements(env, jedges, edges, 0);

	(*env)->ReleaseDoubleArrayElements(env, jterms, terms, 0);

	gst_free_param (params);
	gst_free_metric (metric);

	return smtlength;
}
