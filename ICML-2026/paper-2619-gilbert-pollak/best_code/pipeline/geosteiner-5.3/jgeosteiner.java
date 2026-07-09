/***********************************************************************

	$Id: jgeosteiner.java,v 1.7 2022/11/19 13:45:52 warme Exp $

	File:	jgeosteiner.java
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Example use of Java Native Interface for GeoSteiner.

************************************************************************

	Modification Log:

	a-1:	02/12/2006	warme
		: Added banner and modlog.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

************************************************************************/

/***************************************************************************

The following code is an example of how to use Java Native Interface (JNI)
to get results from GeoSteiner in a Java program.

ToDo:
o Make some kind of wrapper which simply takes a list of Java points.
  The wrapper should convert the points to a list of doubles and then
  call GeoSteiner. The resulting list of doubles for Steiner points should
  be converted to Java points.

o Remove the main-function which is only for debug purposes.

Documentation:
o The lambda value sets the number of legal orientations. The special values
  0 and 1 are the Euclidean and rectilinear metrics, respectively. The latter
  is the same as lambda=2, BUT inside geosteiner they are handled differently
  i.e. different solution methods are applied. Lambda=1 is much faster than
  lambda=2, but the latter is still interesting because the solutions are most
  often not the same (same length but different trees).
o The max_fst_size should default to 0 which is a special value for no
  limitation. Small values will make the solution process much faster, but the
  optimal solution is then not guaranteed to be found.

****************************************************************************/

class jgeosteiner {
	/* Interface to GeoSteiner */
	public native void open();
	public native void close();
	public native double SMT(
		int lambda,		/* IN - number of legal orientations */
		int max_fst_size,	/* IN - largest fst size allowed */
		double[] terminals,	/* IN - terminals */
		int[] nsteiners,	/* OUT - number of Steiner points */
		double[] steiners,	/* OUT - Steiner points */
		int[] nedges,		/* OUT - number of edges */
		int[] edges);		/* OUT - edges (index pairs) */

	/* Make sure that jgeosteiner is in $LD_LIBRARY_PATH */
	static {
		System.loadLibrary("jgeosteiner");
	}

	/* Small debug function allowing one to run 'java jgeosteiner' */
	public static void main(String[] args) {
		/* A small test instance with 4 terminals */
		int nterms = 4;
		double[] terms = {0, 0, 0, 100, 400, 0, 400, 100};

		/* The result is returned in the following variables. Note the
		   trick of using single-element-arrays for number of edges
		   and Steiner points.
		   Probably a nicer solution exists. */
		int[] nedges = {0}; int[] nsteins = {0};
		double[] stps = new double[2*nterms];
		int[] edges = new int[4*nterms];

		/* Create an instance of this class */
		jgeosteiner geo = new jgeosteiner();

		/* Open GeoSteiner. Optimally this should only happen once. */
		geo.open();

		/* Get the Euclidean Steiner minimum tree. */
		double length = geo.SMT(0, 0, terms, nsteins, stps, nedges, edges);

		/* Print the resulting edges and Steiner points. The Steiner
		   points are numbered {nterms, nterms+1, ...} i.e. whenever
		   and edge goes to point i where i<nterms then it is the
		   i'th terminal. Otherwise it is the (i-nterms)'th Steiner
		   point. Confused? */

		for(int i = 0; i<nedges[0]; i++)
		{
			System.out.print("Edge: " + edges[2*i]   + ", "
						  + edges[2*i+1] + "\n");
		}
		for(int i = 0; i<nsteins[0]; i++)
		{
			System.out.print("Stp: " + stps[2*i]   + ", "
						 + stps[2*i+1] + "\n");
		}
		System.out.print("Length: " + length + "\n");

		/* Close GeoSteiner. Optimally this should only happen once. */
		geo.close();
	}
}
