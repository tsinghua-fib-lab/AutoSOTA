/* 11/18/2022: warme: Fix -Wall issues. */

#include <string.h>
#include "lpkit.h"
#include "lpglob.h"
#include "debug.h"

/* Globals used by solver */
static short JustInverted;
static short Status;
static short Doiter;
static short DoInvert;
static short Break_bb;

/* Status values seen only internally to the solver... */
#define	SWITCH_TO_PRIMAL		-1
#define SINGULAR_BASIS			-2
#define	LOST_PRIMAL_FEASIBILITY		-3


static void ftran(lprec *lp, REAL *pcol)
{
  int  i, j, k, r, *rowp;
  REAL theta, *valuep;

  for(i = 1; i <= lp->eta_size; i++) {
    k = lp->eta_col_end[i] - 1;
    r = lp->eta_row_nr[k];
    theta = pcol[r];
    if(theta != 0) {
      j = lp->eta_col_end[i - 1];

      /* CPU intensive loop, let's do pointer arithmetic */
      for(rowp = lp->eta_row_nr + j, valuep = lp->eta_value + j;
	  j < k;
	  j++, rowp++, valuep++)
	pcol[*rowp] += theta * *valuep;

      pcol[r] *= lp->eta_value[k];
    }
  }

  /* round small values to zero */
  for(i = 0; i <= lp->rows; i++)
    my_round(pcol[i], lp->epsel);
} /* ftran */


void btran(lprec *lp, REAL *row)
{
  int  i, j, k, *rowp;
  REAL f, *valuep;

  for(i = lp->eta_size; i >= 1; i--) {
    f = 0;
    k = lp->eta_col_end[i] - 1;
    j = lp->eta_col_end[i - 1];

    for(rowp = lp->eta_row_nr + j, valuep = lp->eta_value + j;
	j <= k;
	j++, rowp++, valuep++)
      f += row[*rowp] * *valuep;

    my_round(f, lp->epsel);
    row[lp->eta_row_nr[k]] = f;
  }
} /* btran */


static void set_row_end(lprec *lp)
{
  int i, j, row_nr;
  int *num, *rownum;

  MALLOC(num, lp->rows + 1);
  MALLOC(rownum, lp->rows + 1);

  for(i = 0; i <= lp->rows; i++) {
    num[i] = 0;
    rownum[i] = 0;
  }

  for(i = 0; i < lp->non_zeros; i++)
    rownum[lp->mat[i].row_nr]++;

  lp->row_end[0] = 0;

  for(i = 1; i <= lp->rows; i++)
    lp->row_end[i] = lp->row_end[i - 1] + rownum[i];

  for(i = 1; i <= lp->columns; i++)
    for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
      row_nr = lp->mat[j].row_nr;
      if(row_nr != 0) {
	num[row_nr]++;
	lp->col_no[lp->row_end[row_nr - 1] + num[row_nr]] = i;
      }
    }

  free(num);
  free(rownum);
  lp->row_end_valid = TRUE;
}


static short isvalid(lprec *lp)
{
  int i, j, *rownum, *colnum;

  if(!lp->row_end_valid) {
    set_row_end(lp);
  }

  if(lp->valid)
    return(TRUE);

  CALLOC(rownum, lp->rows + 1);
  CALLOC(colnum, lp->columns + 1);

  for(i = 1 ; i <= lp->columns; i++)
    for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
      colnum[i]++;
      rownum[lp->mat[j].row_nr]++;
    }

  for(i = 1; i <= lp->columns; i++) {
    if(lp->orig_lowbo[lp->rows + i] == lp->orig_upbo[lp->rows + i]) continue;
    if(colnum[i] == 0) {
      if(lp->names_used)
        printf("%% Warning: Variable %s not used in any constraints\n",
		lp->col_name[i]);
      else
        printf("%% Warning: Variable %d not used in any constraints\n",
		i);
    }
  }
  free(rownum);
  free(colnum);
  lp->valid = TRUE;
  return(TRUE);
}

static void resize_eta(lprec *lp)
{
  lp->eta_alloc *= 1.5;
  REALLOC(lp->eta_value, lp->eta_alloc);
  REALLOC(lp->eta_row_nr, lp->eta_alloc);
  /* printf("%% resized eta to size %d\n", lp->eta_alloc); */
} /* resize_eta */


static void condensecol(lprec *lp,
			int row_nr,
			REAL *pcol)
{
  int i, elnr;

  elnr = lp->eta_col_end[lp->eta_size];

  if(elnr + lp->rows + 2 >= lp->eta_alloc) /* maximum local growth of Eta */
    resize_eta(lp);

  for(i = 0; i <= lp->rows; i++)
    if(i != row_nr && pcol[i] != 0) {
      lp->eta_row_nr[elnr] = i;
      lp->eta_value[elnr] = pcol[i];
      elnr++;
    }

  lp->eta_row_nr[elnr] = row_nr;
  lp->eta_value[elnr] = pcol[row_nr];
  elnr++;
  lp->eta_col_end[lp->eta_size + 1] = elnr;
} /* condensecol */


static void addetacol(lprec *lp)
{
  int  i, j, k;
  REAL theta;

  j = lp->eta_col_end[lp->eta_size];
  lp->eta_size++;
  k = lp->eta_col_end[lp->eta_size] - 1;
  theta = 1 / (REAL) lp->eta_value[k];
  lp->eta_value[k] = theta;
  for(i = j; i < k; i++)
    lp->eta_value[i] *= -theta;
  JustInverted = FALSE;
} /* addetacol */


static void setpivcol(lprec *lp,
		      int   varin,
		      REAL *pcol)
{
  int  i, colnr;

  for(i = 0; i <= lp->rows; i++)
    pcol[i] = 0;

  if(lp->lower[varin]) {
    if(varin > lp->rows) {
      colnr = varin - lp->rows;
      for(i = lp->col_end[colnr - 1]; i < lp->col_end[colnr]; i++)
	pcol[lp->mat[i].row_nr] = lp->mat[i].value;
      pcol[0] -= Extrad;
    }
    else
      pcol[varin] = 1;
  }
  else { /* !lower */
    if(varin > lp->rows) {
      colnr = varin - lp->rows;
      for(i = lp->col_end[colnr - 1]; i < lp->col_end[colnr]; i++)
	pcol[lp->mat[i].row_nr] = -lp->mat[i].value;
      pcol[0] += Extrad;
    }
    else
      pcol[varin] = -1;
  }

  ftran(lp, pcol);
} /* setpivcol */


static void minoriteration(lprec *lp,
			   int colnr,
			   int row_nr)
{
  int  i, j, k, wk, varin, varout, elnr;
  REAL piv = 0, theta;

  varin = colnr + lp->rows;
  elnr = lp->eta_col_end[lp->eta_size];
  wk = elnr;
  lp->eta_size++;

  if(Extrad != 0) {
    lp->eta_row_nr[elnr] = 0;
    lp->eta_value[elnr] = -Extrad;
    elnr++;
    if(elnr >= lp->eta_alloc)
      resize_eta(lp);
  }

  for(j = lp->col_end[colnr - 1] ; j < lp->col_end[colnr]; j++) {
    k = lp->mat[j].row_nr;

    if(k == 0 && Extrad != 0)
      lp->eta_value[lp->eta_col_end[lp->eta_size - 1]] += lp->mat[j].value;
    else if(k != row_nr) {
      lp->eta_row_nr[elnr] = k;
      lp->eta_value[elnr] = lp->mat[j].value;
      elnr++;
      if(elnr >= lp->eta_alloc)
	resize_eta(lp);
    }
    else
      piv = lp->mat[j].value;
  }

  lp->eta_row_nr[elnr] = row_nr;
  lp->eta_value[elnr] = 1 / piv;
  theta = lp->rhs[row_nr] / piv;
  lp->rhs[row_nr] = theta;

  for(i = wk; i < elnr; i++)
    lp->rhs[lp->eta_row_nr[i]] -= theta * lp->eta_value[i];

  varout = lp->bas[row_nr];
  lp->bas[row_nr] = varin;
  lp->basis[varout] = FALSE;
  lp->basis[varin] = TRUE;

  for(i = wk; i < elnr; i++)
    lp->eta_value[i] /= -piv;

  ++elnr;
  if(elnr >= lp->eta_alloc)
    resize_eta(lp);
  lp->eta_col_end[lp->eta_size] = elnr;
} /* minoriteration */


static void rhsmincol(lprec *lp,
		      REAL theta,
		      int row_nr,
		      int varin)
{
  int  i, j, k, varout;
  REAL f;

  if(row_nr > lp->rows + 1) {
    printf("%% Error: rhsmincol called with row_nr: %d, rows: %d\n",
	    row_nr, lp->rows);
    printf("%% This indicates numerical instability\n");
    exit(EXIT_FAILURE);
  }

  j = lp->eta_col_end[lp->eta_size];
  k = lp->eta_col_end[lp->eta_size + 1];
  for(i = j; i < k; i++) {
    f = lp->rhs[lp->eta_row_nr[i]] - theta * lp->eta_value[i];
    my_round(f, lp->epsb);
    lp->rhs[lp->eta_row_nr[i]] = f;
  }

  lp->rhs[row_nr] = theta;
  varout = lp->bas[row_nr];
  lp->bas[row_nr] = varin;
  lp->basis[varout] = FALSE;
  lp->basis[varin] = TRUE;
} /* rhsmincol */


short invert(lprec *lp)
{
  int    i, j, v, wk, numit, varnr, row_nr, colnr, varin;
  REAL   theta;
  REAL   *pcol;
  int	 singularities;
  short  *frow;
  short  *fcol;
  int    *rownum, *col, *row;
  int    *colnum;

  if(lp->print_at_invert)
    printf("%% Start Invert iter %d eta_size %d rhs[0] %g \n",
	    lp->iter, lp->eta_size, (double) - lp->rhs[0]);

  CALLOC(rownum, lp->rows + 1);
  CALLOC(col, lp->rows + 1);
  CALLOC(row, lp->rows + 1);
  CALLOC(pcol, lp->rows + 1);
  CALLOC(frow, lp->rows + 1);
  CALLOC(fcol, lp->columns + 1);
  CALLOC(colnum, lp->columns + 1);

  for(i = 0; i <= lp->rows; i++)
    frow[i] = TRUE;

  for(i = 0; i < lp->columns; i++)
    fcol[i] = FALSE;

  for(i = 0; i < lp->rows; i++)
    rownum[i] = 0;

  for(i = 0; i <= lp->columns; i++)
    colnum[i] = 0;

  for(i = 0; i <= lp->rows; i++)
    if(lp->bas[i] > lp->rows)
      fcol[lp->bas[i] - lp->rows - 1] = TRUE;
    else
      frow[lp->bas[i]] = FALSE;

  for(i = 1; i <= lp->rows; i++)
    if(frow[i])
      for(j = lp->row_end[i - 1] + 1; j <= lp->row_end[i]; j++) {
	wk = lp->col_no[j];
	if(fcol[wk - 1]) {
	  colnum[wk]++;
	  rownum[i - 1]++;
	}
      }

  for(i = 1; i <= lp->rows; i++)
    lp->bas[i] = i;

  for(i = 1; i <= lp->rows; i++)
    lp->basis[i] = TRUE;

  for(i = 1; i <= lp->columns; i++)
    lp->basis[i + lp->rows] = FALSE;

  for(i = 0; i <= lp->rows; i++)
    lp->rhs[i] = lp->rh[i];

  for(i = 1; i <= lp->columns; i++) {
    varnr = lp->rows + i;
    if(!lp->lower[varnr]) {
      theta = lp->upbo[varnr];
      for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
	lp->rhs[lp->mat[j].row_nr] -= theta * lp->mat[j].value;
    }
  }

  for(i = 1; i <= lp->rows; i++)
    if(!lp->lower[i])
      lp->rhs[i] -= lp->upbo[i];

  singularities = 0;
  lp->eta_size = 0;
  v = 0;
  row_nr = 0;
  lp->num_inv = 0;
  numit = 0;

  while(v < lp->rows) {
    row_nr++;
    if(row_nr > lp->rows)
      row_nr = 1;

    v++;

    if(rownum[row_nr - 1] == 1)
      if(frow[row_nr]) {
	v = 0;
	j = lp->row_end[row_nr - 1] + 1;

	while(!(fcol[lp->col_no[j] - 1]))
	  j++;

	colnr = lp->col_no[j];
	fcol[colnr - 1] = FALSE;
	colnum[colnr] = 0;

	for(j = lp->col_end[colnr - 1]; j < lp->col_end[colnr]; j++)
	  if(frow[lp->mat[j].row_nr])
	    rownum[lp->mat[j].row_nr - 1]--;

	frow[row_nr] = FALSE;
	minoriteration(lp, colnr, row_nr);
      }
  }
  v = 0;
  colnr = 0;
  while(v < lp->columns) {
    colnr++;
    if(colnr > lp->columns)
      colnr = 1;

    v++;

    if(colnum[colnr] == 1)
      if(fcol[colnr - 1]) {
	v = 0;
	j = lp->col_end[colnr - 1] + 1;

	while(!(frow[lp->mat[j - 1].row_nr]))
	  j++;

	row_nr = lp->mat[j - 1].row_nr;
	frow[row_nr] = FALSE;
	rownum[row_nr - 1] = 0;

	for(j = lp->row_end[row_nr - 1] + 1; j <= lp->row_end[row_nr]; j++)
	  if(fcol[lp->col_no[j] - 1])
	    colnum[lp->col_no[j]]--;

	fcol[colnr - 1] = FALSE;
	numit++;
	col[numit - 1] = colnr;
	row[numit - 1] = row_nr;
      }
  }
  for(j = 1; j <= lp->columns; j++)
    if(fcol[j - 1]) {
      fcol[j - 1] = FALSE;
      setpivcol(lp, j + lp->rows, pcol);
      row_nr = 1;

      while((row_nr <= lp->rows) && (!(frow[row_nr] && pcol[row_nr])))
	row_nr++;

      /* if(row_nr == lp->rows + 1) */
      if(row_nr > lp->rows) {
	/* This column is singular!  Just skip it, leaving one of the */
	/* slack variables basic in its place... */
	printf("%% Column %d singular!\n", j);
	++singularities;
      }
      else {
	frow[row_nr] = FALSE;
	condensecol(lp, row_nr, pcol);
	theta = lp->rhs[row_nr] / (REAL) pcol[row_nr];
	rhsmincol(lp, theta, row_nr, lp->rows + j);
	addetacol(lp);
      }
    }

  for(i = numit - 1; i >= 0; i--) {
    colnr = col[i];
    row_nr = row[i];
    varin = colnr + lp->rows;

    for(j = 0; j <= lp->rows; j++)
      pcol[j] = 0;

    for(j = lp->col_end[colnr - 1]; j < lp->col_end[colnr]; j++)
      pcol[lp->mat[j].row_nr] = lp->mat[j].value;

    pcol[0] -= Extrad;
    condensecol(lp, row_nr, pcol);
    theta = lp->rhs[row_nr] / (REAL) pcol[row_nr];
    rhsmincol(lp, theta, row_nr, varin);
    addetacol(lp);
  }

  for(i = 1; i <= lp->rows; i++)
    my_round(lp->rhs[i], lp->epsb);

  if(lp->print_at_invert)
    printf("%% End Invert                eta_size %d rhs[0] %g\n",
	    lp->eta_size, (double) - lp->rhs[0]);

  JustInverted = TRUE;
  DoInvert = FALSE;
  free(rownum);
  free(col);
  free(row);
  free(pcol);
  free(frow);
  free(fcol);
  free(colnum);

  return (singularities <= 0);
} /* invert */

static int colprim(lprec *lp,
		   short minit,
		   REAL   *drow)
{
  int  varnr, i, j, colnr;
  REAL f, dpiv;

  dpiv = -lp->epsd;
  colnr = 0;
  if(!minit) {
    for(i = 1; i <= lp->sum; i++)
      drow[i] = 0;
    drow[0] = 1;
    btran(lp, drow);
    for(i = 1; i <= lp->columns; i++) {
      varnr = lp->rows + i;
      if(!lp->basis[varnr])
	if(lp->upbo[varnr] > 0) {
	  f = 0;
	  for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
	    f += drow[lp->mat[j].row_nr] * lp->mat[j].value;
	  drow[varnr] = f;
	}
    }
    for(i = 1; i <= lp->sum; i++)
      my_round(drow[i], lp->epsd);
  }
  for(i = 1; i <= lp->sum; i++)
    if(!lp->basis[i])
      if(lp->upbo[i] > 0) {
	if(lp->lower[i])
	  f = drow[i];
	else
	  f = -drow[i];
	if(f < dpiv) {
	  dpiv = f;
	  colnr = i;
	}
      }
  if(lp->trace) {
    if(colnr > 0)
      printf("%% col_prim:%d, reduced cost: %g\n",
	      colnr, (double)dpiv);
    else
      printf("%% col_prim: no negative reduced costs found, optimality!\n");
  }
  if(colnr == 0) {
    Doiter   = FALSE;
    DoInvert = FALSE;
    Status   = OPTIMAL;
  }
  return(colnr);
} /* colprim */

static int rowprim(lprec *lp,
		   int colnr,
		   REAL *theta,
		   REAL *pcol)
{
  int  i, row_nr;
  REAL f, quot;

  row_nr = 0;
  (*theta) = lp->infinite;
  f = 0.0;
  for(i = 1; i <= lp->rows; i++) {
    f = pcol[i];
    if(f != 0) {
      if(my_abs(f) < Trej) {
	debug_print(lp, "pivot %g rejected, too small (limit %g)\n",
		    (double)f, (double)Trej);
      }
      else { /* pivot alright */
	quot = 2 * lp->infinite;
	if(f > 0)
	  quot = lp->rhs[i] / (REAL) f;
	else if(lp->upbo[lp->bas[i]] < lp->infinite)
	  quot = (lp->rhs[i] - lp->upbo[lp->bas[i]]) / (REAL) f;
	my_round(quot, lp->epsel);
	if(quot < (*theta)) {
	  (*theta) = quot;
	  row_nr = i;
	}
      }
    }
  }
  if(row_nr == 0)
    for(i = 1; i <= lp->rows; i++) {
      f = pcol[i];
      if(f != 0) {
	quot = 2 * lp->infinite;
	if(f > 0)
	  quot = lp->rhs[i] / (REAL) f;
	else
	  if(lp->upbo[lp->bas[i]] < lp->infinite)
	    quot = (lp->rhs[i] - lp->upbo[lp->bas[i]]) / (REAL) f;
	my_round(quot, lp->epsel);
	if(quot < (*theta)) {
	  (*theta) = quot;
	  row_nr = i;
	}
      }
    }

  if((*theta) < 0) {
    printf("%% Warning: Numerical instability, qout = %g\n",
	    (double)(*theta));
    printf("%% pcol[%d] = %18g, rhs[%d] = %18g , upbo = %g\n",
	    row_nr, (double)f, row_nr, (double)lp->rhs[row_nr],
	    (double)lp->upbo[lp->bas[row_nr]]);
  }
  if(row_nr == 0) {
    if(lp->upbo[colnr] == lp->infinite) {
      Doiter   = FALSE;
      DoInvert = FALSE;
      Status   = UNBOUNDED;
    }
    else {
      i = 1;
      while(pcol[i] >= 0 && i <= lp->rows)
	i++;
      if(i > lp->rows) { /* empty column with upperbound! */
	lp->lower[colnr] = FALSE;
	lp->rhs[0] += lp->upbo[colnr]*pcol[0];
	Doiter = FALSE;
	DoInvert = FALSE;
      }
      else if(pcol[i]<0) {
	row_nr = i;
      }
    }
  }
  if(row_nr > 0)
    Doiter = TRUE;
  if(lp->trace)
    printf("%% row_prim:%d, pivot element:%18g\n", row_nr,
	    (double)pcol[row_nr]);

  return(row_nr);
} /* rowprim */

static int rowdual(lprec *lp)
{
  int   i, row_nr;
  REAL  f, g, minrhs;
  short artifs;

  row_nr = 0;
  minrhs = -lp->epsb;
  i = 0;
  artifs = FALSE;
  while(i < lp->rows && !artifs) {
    i++;
    f = lp->upbo[lp->bas[i]];
    if(f == 0 && (lp->rhs[i] != 0)) {
      artifs = TRUE;
      row_nr = i;
    }
    else {
      if(lp->rhs[i] < f - lp->rhs[i])
	g = lp->rhs[i];
      else
	g = f - lp->rhs[i];
      if(g < minrhs) {
	minrhs = g;
	row_nr = i;
      }
    }
  }

  if(lp->trace) {
    if(row_nr > 0) {
      printf("%% row_dual:%d, rhs of selected row:           %18g\n",
	      row_nr, (double)lp->rhs[row_nr]);
      if(lp->upbo[lp->bas[row_nr]] < lp->infinite)
	printf("%%\t\tupper bound of basis variable:    %18g\n",
		(double)lp->upbo[lp->bas[row_nr]]);
    }
    else
      printf("%% row_dual: no infeasibilities found\n");
  }

  return(row_nr);
} /* rowdual */

static int coldual(lprec *lp,
		   int row_nr,
		   short minit,
		   REAL *prow,
		   REAL *drow)
{
  int  i, j, k, r, varnr, *rowp, row, colnr;
  REAL theta, quot, pivot, d, f, g, *valuep, value;

  Doiter = FALSE;
  if(!minit) {
    for(i = 0; i <= lp->rows; i++) {
      prow[i] = 0;
      drow[i] = 0;
    }

    drow[0] = 1;
    prow[row_nr] = 1;

    for(i = lp->eta_size; i >= 1; i--) {
      d = 0;
      f = 0;
      k = lp->eta_col_end[i] - 1;
      r = lp->eta_row_nr[k];
      j = lp->eta_col_end[i - 1];

      /* this is one of the loops where the program consumes a lot of CPU
	 time */
      /* let's help the compiler by doing some pointer arithmetic instead
	 of array indexing */
      for(rowp = lp->eta_row_nr + j, valuep = lp->eta_value + j;
	  j <= k;
	  j++, rowp++, valuep++) {
	f += prow[*rowp] * *valuep;
	d += drow[*rowp] * *valuep;
      }

      my_round(f, lp->epsel);
      prow[r] = f;
      my_round(d, lp->epsd);
      drow[r] = d;
    }

    for(i = 1; i <= lp->columns; i++) {
      varnr = lp->rows + i;
      if(!lp->basis[varnr]) {
	matrec *matentry;

	d = - Extrad * drow[0];
	f = 0;
	k = lp->col_end[i];
	j = lp->col_end[i - 1];

	/* this is one of the loops where the program consumes a lot
	   of cpu time */
	/* let's help the compiler with pointer arithmetic instead
	   of array indexing */
	for(matentry = lp->mat + j;
	    j < k;
	    j++, matentry++) {
	  row = (*matentry).row_nr;
	  value = (*matentry).value;
	  d += drow[row] * value;
	  f += prow[row] * value;
	}

	my_round(f, lp->epsel);
	prow[varnr] = f;
	my_round(d, lp->epsd);
	drow[varnr] = d;
      }
    }
  }

  if(lp->rhs[row_nr] > lp->upbo[lp->bas[row_nr]])
    g = -1;
  else
    g = 1;

  pivot = 0;
  colnr = 0;
  theta = lp->infinite;

  for(i = 1; i <= lp->sum; i++) {
    if(lp->lower[i])
      d = prow[i] * g;
    else
      d = -prow[i] * g;

    if((d < 0) && (!lp->basis[i]) && (lp->upbo[i] > 0)) {
      if(lp->lower[i])
	quot = -drow[i] / (REAL) d;
      else
	quot = drow[i] / (REAL) d;
      if(quot < theta) {
	theta = quot;
	pivot = d;
	colnr = i;
      }
      else if((quot == theta) && (my_abs(d) > my_abs(pivot))) {
	pivot = d;
	colnr = i;
      }
    }
  }

  if(lp->trace)
    printf("%% col_dual:%d, pivot element:  %18g\n", colnr,
	    (double)prow[colnr]);

  if(colnr > 0)
    Doiter = TRUE;

  return(colnr);
} /* coldual */

static void iteration(lprec *lp,
		      int row_nr,
		      int varin,
		      REAL *theta,
		      REAL up,
		      short *minit,
		      short *low,
		      short primal,
                      REAL *pcol)
{
  int  i, k, varout;
  REAL f;
  REAL pivot;

  lp->iter++;

  if(((*minit) = (*theta) > (up + lp->epsb))) {
    (*theta) = up;
    (*low) = !(*low);
  }

  k = lp->eta_col_end[lp->eta_size + 1];
  pivot = lp->eta_value[k - 1];

  for(i = lp->eta_col_end[lp->eta_size]; i < k; i++) {
    f = lp->rhs[lp->eta_row_nr[i]] - (*theta) * lp->eta_value[i];
    my_round(f, lp->epsb);
    lp->rhs[lp->eta_row_nr[i]] = f;
  }

  if(!(*minit)) {
    lp->rhs[row_nr] = (*theta);
    varout = lp->bas[row_nr];
    lp->bas[row_nr] = varin;
    lp->basis[varout] = FALSE;
    lp->basis[varin] = TRUE;

    if(primal && pivot < 0)
      lp->lower[varout] = FALSE;

    if(!(*low) && up < lp->infinite) {
      (*low) = TRUE;
      lp->rhs[row_nr] = up - lp->rhs[row_nr];
      for(i = lp->eta_col_end[lp->eta_size]; i < k; i++)
	lp->eta_value[i] = -lp->eta_value[i];
    }

    addetacol(lp);
    lp->num_inv++;
  }

  if(lp->trace) {
    printf("%% Theta = %g ", (double)(*theta));
    if((*minit)) {
      if(!lp->lower[varin])
	printf("%% Iteration: %d, variable %d changed from 0 to its upper bound of %g\n",
		lp->iter, varin, (double)lp->upbo[varin]);
      else
	printf("%% Iteration: %d, variable %d changed its upper bound of %g to 0\n",
		lp->iter, varin, (double)lp->upbo[varin]);
    }
    else
      printf("%% Iteration: %d, variable %d entered basis at: %g\n",
	      lp->iter, varin, (double)lp->rhs[row_nr]);
    if(!primal) {
      f = 0;
      for(i = 1; i <= lp->rows; i++)
	if(lp->rhs[i] < 0)
	  f -= lp->rhs[i];
	else
	  if(lp->rhs[i] > lp->upbo[lp->bas[i]])
	    f += lp->rhs[i] - lp->upbo[lp->bas[i]];
      printf("%% feasibility gap of this basis: %g\n",
	      (double)f);
    }
    else
      printf("%% objective function value of this feasible basis: %g\n",
	      (double)lp->rhs[0]);
  }
} /* iteration */


static void primloop(lprec *lp)
{
  int	 i;
  REAL   theta, x;
  short  primal;
  REAL   *drow, *prow, *Pcol;
  short  minit;
  int    colnr, row_nr;

  if(lp->trace)
    printf("%% Entering primal algorithm\n");

  CALLOC(drow, lp->sum + 1);
  CALLOC(prow, lp->sum + 1);
  CALLOC(Pcol, lp->rows + 1);

  Status = RUNNING;
  primal = TRUE;
  DoInvert = FALSE;
  Doiter = FALSE;
  Extrad = 0;

  row_nr = 0;
  colnr = 0;

  minit = FALSE;

  while(Status == RUNNING) {
    Doiter = FALSE;
    DoInvert = FALSE;

    colnr = colprim(lp, minit, drow);
    if(colnr > 0) {
      setpivcol(lp, colnr, Pcol);

      row_nr = rowprim(lp, colnr, &theta, Pcol);
      if(row_nr > 0)
	condensecol(lp, row_nr, Pcol);
    }

    if(Doiter) {
      iteration(lp, row_nr, colnr, &theta, lp->upbo[colnr], &minit,
		&lp->lower[colnr], primal, Pcol);
    }

    if(lp->num_inv >= lp->max_num_inv)
      DoInvert = TRUE;

    if(DoInvert) {
      if (!invert(lp)) {
	Status = SINGULAR_BASIS;
	break;
      }
      /* Check whether we are still feasible or not... */
      for(i = 1; i <= lp->rows; i++) {
	x = lp->rhs[i];
	if ((x < -lp->epsb) || (x > lp->upbo[lp->bas[i]] + lp->epsb)) {
	  Status = LOST_PRIMAL_FEASIBILITY;
	  break;
	}
      }
    }
  }

  free(drow);
  free(prow);
  free(Pcol);
} /* primloop */


static void dualloop(lprec *lp)
{
  int    i, j;
  REAL   f, theta;
  short  primal;
  REAL   *drow, *prow, *Pcol;
  short  minit;
  int    colnr, row_nr;

  if(lp->trace)
    printf("%% Entering dual algorithm\n");

  CALLOC(drow, lp->sum + 1);
  CALLOC(prow, lp->sum + 1);
  CALLOC(Pcol, lp->rows + 1);

  Status = RUNNING;
  primal = FALSE;
  DoInvert = FALSE;
  Doiter = FALSE;

  /* Set Extrad to be the most negative of the objective coefficients.	*/
  /* We effectively subtract Extrad from every element of the objective	*/
  /* row, thereby making the entire objective row non-negative.  Note	*/
  /* that this forces dual feasibility!  Although it also alters the	*/
  /* objective function, we don't really care about that too much	*/
  /* because we only use the dual algorithm to obtain a primal feasible	*/
  /* solution that we can start the primal algorithm with.  Virtually	*/
  /* any non-zero objective function will work for this!		*/
  Extrad = 0;
  for(i = 1; i <= lp->columns; i++) {
    f = 0;
    for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
      if(lp->mat[j].row_nr == 0)
	f += lp->mat[j].value;

    if(f < Extrad)
      Extrad = f;
  }

  if(lp->trace)
    printf("%% Extrad = %g\n", (double)Extrad);

  row_nr = 0;
  colnr = 0;

  minit = FALSE;

  while(Status == RUNNING) {
    Doiter = FALSE;
    DoInvert = FALSE;

    if(!minit)
      row_nr = rowdual(lp);

    if(row_nr > 0 ) {
      colnr = coldual(lp, row_nr, minit, prow, drow);
      if(colnr > 0) {
	setpivcol(lp, colnr, Pcol);

	/* getting div by zero here. Catch it and try to recover */
	if(Pcol[row_nr] == 0) {
	  printf("%% An attempt was made to divide by zero (Pcol[%d])\n",
		 row_nr);
	  printf("%% This indicates numerical instability\n");
#if 0
	  printf("%%\tLower[%d] = %d\n"
		 "%%\tprow[%d] = %g\n"
		 "%%\tdrow[%d] = %g\n"
		 "%%\tPcol[%d] = %g\n"
		 "%%\tRhs[%d] = %g\n"
		 "%%\tBas[%d] = %d\n"
		 "%%\tUpbo[%d] = %g\n",
		  colnr, Lower[colnr],
		  colnr, prow[colnr],
		  colnr, drow[colnr],
		  row_nr, Pcol[row_nr],
		  row_nr, Rhs[row_nr],
		  row_nr, Bas[row_nr],
		  Bas[row_nr], Upbo[Bas[row_nr]]);
#if 1
	  for (i = 1; i <= lp->sum; i++)
	    if (prow[i] != 0.0)
	      printf("%%\t\tprow[%d] = %g\n", i, prow[i]);
	  printf ("%%");
	  for (i = 1; i <= lp->sum; i++) {
	    printf(" %d/%g", Lower[i], drow[i]);
	    if ((i % 10) == 0)
	      printf("\n%%");
	  }
	  printf("\n");
#endif
#endif
	  Doiter = FALSE;
	  if(!JustInverted) {
	    printf("%% Trying to recover. Reinverting Eta\n");
	    DoInvert = TRUE;
	  }
	  else {
	    printf("%% Can't reinvert, failure\n");
	    dump_lp (lp, "binary.lp");
	    Status = FAILURE;
	  }
	}
	else {
	  condensecol(lp, row_nr, Pcol);
	  f = lp->rhs[row_nr] - lp->upbo[lp->bas[row_nr]];

	  if(f > 0) {
	    theta = f / (REAL) Pcol[row_nr];
	    if(theta <= lp->upbo[colnr] + lp->epsb)
	      lp->lower[lp->bas[row_nr]] = !lp->lower[lp->bas[row_nr]];
	  }
	  else /* f <= 0 */
	    theta = lp->rhs[row_nr] / (REAL) Pcol[row_nr];
	}
      }
      else
	Status = INFEASIBLE;
    }
    else {
      Status   = SWITCH_TO_PRIMAL;
      Doiter   = FALSE;
      Extrad   = 0;
      DoInvert = TRUE;
    }

    if(Doiter)
      iteration(lp, row_nr, colnr, &theta, lp->upbo[colnr], &minit,
		&lp->lower[colnr], primal, Pcol);

    if(lp->num_inv >= lp->max_num_inv)
      DoInvert = TRUE;

    if(DoInvert) {
      if(!invert(lp)) {
	Status = SINGULAR_BASIS;
	break;
      }
    }
  }

  free(drow);
  free(prow);
  free(Pcol);
}


static int solvelp(lprec *lp)
{
  int    i, singular_count, lost_feas_count;
  short  feasible;
  REAL	 x;

  if(lp->do_presolve)
    presolve(lp);

  lp->iter = 0;

  singular_count = 0;
  lost_feas_count = 0;

  for(;;) {
    /* Check whether we are feasible or infeasible. */
    feasible = TRUE;
    for(i = 1; i <= lp->rows; i++) {
      x = lp->rhs[i];
      if((x < 0) || (x > lp->upbo[lp->bas[i]])) {
	feasible = FALSE;
	break;
      }
    }

    if(feasible) {
      primloop(lp);
    }
    else {
      dualloop(lp);
      if(Status == SWITCH_TO_PRIMAL)
	primloop(lp);
    }

    if(Status == SINGULAR_BASIS) {
      ++singular_count;
      if(singular_count >= 5) {
	printf("%% SINGULAR BASIS!  Too many singularities - aborting.\n");
	Status = FAILURE;
	break;
      }
      printf("%% SINGULAR BASIS!  Will attempt to recover.\n");
      /* Singular pivots are simply skipped by the inversion, leaving */
      /* a row's slack var in the basis instead of the singular problem */
      /* var.  This basis could be feasible or infeasible.  Check how to */
      /* restart. */
      continue;
    }
    else if(Status == LOST_PRIMAL_FEASIBILITY) {
      ++lost_feas_count;
      if(lost_feas_count >= 5) {
	printf("%% LOST PRIMAL FEASIBILITY too many times, aborting.\n");
	Status = FAILURE;
	break;
      }
      printf("%% LOST PRIMAL FEASIBILITY!  Recovering.\n");
      continue;
    }
    else {
      /* One of the "normal" status codes -- we are done! */
      break;
    }
  }

  lp->total_iter += lp->iter;

  return(Status);
} /* solvelp */


static short is_int(lprec *lp, int i)
{
  REAL   value, error;

  value = lp->solution[i];
  error = value - (REAL)floor((double)value);

  if(error < lp->epsilon)
    return(TRUE);

  if(error > (1 - lp->epsilon))
    return(TRUE);

  return(FALSE);
} /* is_int */


static void construct_solution(lprec *lp)
{
  int    i, j, basi;
  REAL   f;

  /* zero all results of rows */
  memset(lp->solution, '\0', (lp->rows + 1) * sizeof(REAL));

  lp->solution[0] = -lp->orig_rh[0];

  if(lp->scaling_used) {
    lp->solution[0] /= lp->scale[0];

    for(i = lp->rows + 1; i <= lp->sum; i++)
      lp->solution[i] = lp->lowbo[i] * lp->scale[i];

    for(i = 1; i <= lp->rows; i++) {
      basi = lp->bas[i];
      if(basi > lp->rows)
	lp->solution[basi] += lp->rhs[i] * lp->scale[basi];
    }
    for(i = lp->rows + 1; i <= lp->sum; i++)
      if(!lp->basis[i] && !lp->lower[i])
	lp->solution[i] += lp->upbo[i] * lp->scale[i];

    for(j = 1; j <= lp->columns; j++) {
      f = lp->solution[lp->rows + j];
      if(f != 0)
	for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
	  lp->solution[lp->mat[i].row_nr] += (f / lp->scale[lp->rows+j])
	    * (lp->mat[i].value / lp->scale[lp->mat[i].row_nr]);
    }

    for(i = 0; i <= lp->rows; i++) {
      if(my_abs(lp->solution[i]) < lp->epsb)
	lp->solution[i] = 0;
      else if(lp->ch_sign[i])
	lp->solution[i] = -lp->solution[i];
    }
  }
  else { /* no scaling */
    for(i = lp->rows + 1; i <= lp->sum; i++)
      lp->solution[i] = lp->lowbo[i];

    for(i = 1; i <= lp->rows; i++) {
      basi = lp->bas[i];
      if(basi > lp->rows)
	lp->solution[basi] += lp->rhs[i];
    }

    for(i = lp->rows + 1; i <= lp->sum; i++)
      if(!lp->basis[i] && !lp->lower[i])
	lp->solution[i] += lp->upbo[i];

    for(j = 1; j <= lp->columns; j++) {
      f = lp->solution[lp->rows + j];
      if(f != 0)
	for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
	  lp->solution[lp->mat[i].row_nr] += f * lp->mat[i].value;
    }

    for(i = 0; i <= lp->rows; i++) {
      if(my_abs(lp->solution[i]) < lp->epsb)
	lp->solution[i] = 0;
      else if(lp->ch_sign[i])
	lp->solution[i] = -lp->solution[i];
    }
  }
} /* construct_solution */

static void calculate_duals(lprec *lp)
{
  int i;

  /* initialize */
  lp->duals[0] = 1;
  for(i = 1; i <= lp->rows; i++)
    lp->duals[i] = 0;

  btran(lp, lp->duals);

  if(lp->scaling_used)
    for(i = 1; i <= lp->rows; i++)
      lp->duals[i] *= lp->scale[i] / lp->scale[0];

  /* the dual values are the reduced costs of the slacks */
  /* When the slack is at its upper bound, change the sign. */
  for(i = 1; i <= lp->rows; i++) {
    if(lp->basis[i])
      lp->duals[i] = 0;
    /* added a test if variable is different from 0 because sometime you get
       -0 and this is different from 0 on for example INTEL processors (ie 0
       != -0 on INTEL !) PN */
    else if((lp->ch_sign[0] == lp->ch_sign[i]) && lp->duals[i])
      lp->duals[i] = - lp->duals[i];
  }
} /* calculate_duals */

static void check_if_less(REAL x,
			  REAL y,
			  REAL value)
{
  if(x >= y) {
    printf("%% Error: new upper or lower bound is not more restrictive\n");
    printf("%% bound 1: %g, bound 2: %g, value: %g\n",
	    (double)x, (double)y, (double)value);
    /* exit(EXIT_FAILURE); */
  }
}

static void check_solution(lprec *lp,
			   REAL *upbo,
			   REAL *lowbo)
{
  int i;

  /* check if all solution values are within the bounds, but allow some margin
     for numerical errors */

#define CHECK_EPS 1e-2

  if(lp->columns_scaled)
    for(i = lp->rows + 1; i <= lp->sum; i++) {
      if(lp->solution[i] < lowbo[i] * lp->scale[i] - CHECK_EPS) {
	printf("%% Error: variable %d (%s) has a solution (%g) smaller than its lower bound (%g)\n",
		i - lp->rows, lp->col_name[i - lp->rows],
		(double)lp->solution[i], (double)lowbo[i] * lp->scale[i]);
	/* abort(); */
      }

      if(lp->solution[i] > upbo[i] * lp->scale[i] + CHECK_EPS) {
	printf("%% Error: variable %d (%s) has a solution (%g) larger than its upper bound (%g)\n",
		i - lp->rows, lp->col_name[i - lp->rows],
		(double)lp->solution[i], (double)upbo[i] * lp->scale[i]);
	/* abort(); */
      }
    }
  else /* columns not scaled */
    for(i = lp->rows + 1; i <= lp->sum; i++) {
      if(lp->solution[i] < lowbo[i] - CHECK_EPS) {
	printf("%% Error: variable %d (%s) has a solution (%g) smaller than its lower bound (%g)\n",
		i - lp->rows, lp->col_name[i - lp->rows],
		(double)lp->solution[i], (double)lowbo[i]);
	/* abort(); */
      }

      if(lp->solution[i] > upbo[i] + CHECK_EPS) {
	printf("%% Error: variable %d (%s) has a solution (%g) larger than its upper bound (%g)\n",
		i - lp->rows, lp->col_name[i - lp->rows],
		(double)lp->solution[i], (double)upbo[i]);
	/* abort(); */
      }
    }
} /* check_solution */


static int milpsolve(lprec *lp,
		     REAL   *upbo,
		     REAL   *lowbo,
		     short  *sbasis,
		     short  *slower,
		     int    *sbas,
		     int     recursive)
{
  int i, j, failure, notint, is_worse;
  REAL theta, tmpreal;

  (void) check_solution;

  if(Break_bb)
    return(BREAK_BB);

  Level++;
  lp->total_nodes++;

  if(Level > lp->max_level)
    lp->max_level = Level;

  debug_print(lp, "starting solve");

  /* make fresh copies of upbo, lowbo, rh as solving changes them */
  memcpy(lp->upbo,  upbo,    (lp->sum + 1)  * sizeof(REAL));
  memcpy(lp->lowbo, lowbo,   (lp->sum + 1)  * sizeof(REAL));
  memcpy(lp->rh,    lp->orig_rh, (lp->rows + 1) * sizeof(REAL));

  /* make shure we do not do memcpy(lp->basis, lp->basis ...) ! */
  if(recursive) {
    memcpy(lp->basis, sbasis,  (lp->sum + 1)  * sizeof(short));
    memcpy(lp->lower, slower,  (lp->sum + 1)  * sizeof(short));
    memcpy(lp->bas,   sbas,    (lp->rows + 1) * sizeof(int));
  }

  if(lp->anti_degen) { /* randomly disturb bounds */
    for(i = 1; i <= lp->columns; i++) {
      tmpreal = (REAL) (rand() % 100 * 0.00001);
      if(tmpreal > lp->epsb)
	lp->lowbo[i + lp->rows] -= tmpreal;
      tmpreal = (REAL) (rand() % 100 * 0.00001);
      if(tmpreal > lp->epsb)
	lp->upbo[i + lp->rows] += tmpreal;
    }
    lp->eta_valid = FALSE;
  }

  if(!lp->eta_valid) {
    /* transform to all lower bounds to zero */
    for(i = 1; i <= lp->columns; i++)
      if((theta = lp->lowbo[lp->rows + i]) != 0) {
	if(lp->upbo[lp->rows + i] < lp->infinite)
	  lp->upbo[lp->rows + i] -= theta;
	for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
	  lp->rh[lp->mat[j].row_nr] -= theta * lp->mat[j].value;
      }
    invert(lp);
    lp->eta_valid = TRUE;
  }

  failure = solvelp(lp);

  if(lp->anti_degen && (failure == OPTIMAL)) {
    /* restore to original problem, solve again starting from the basis found
       for the disturbed problem */

    /* restore original problem */
    memcpy(lp->upbo,  upbo,        (lp->sum + 1)  * sizeof(REAL));
    memcpy(lp->lowbo, lowbo,       (lp->sum + 1)  * sizeof(REAL));
    memcpy(lp->rh,    lp->orig_rh, (lp->rows + 1) * sizeof(REAL));

    /* transform to all lower bounds zero */
    for(i = 1; i <= lp->columns; i++)
      if((theta = lp->lowbo[lp->rows + i] != 0)) {
	if(lp->upbo[lp->rows + i] < lp->infinite)
	  lp->upbo[lp->rows + i] -= theta;
	for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
	  lp->rh[lp->mat[j].row_nr] -= theta * lp->mat[j].value;
      }
    invert(lp);
    lp->eta_valid = TRUE;
    failure = solvelp(lp); /* and solve again */
  }

  if(failure != OPTIMAL)
    debug_print(lp, "this problem has no solution, it is %s",
		(failure == UNBOUNDED) ? "unbounded" : "infeasible");

  if(failure == INFEASIBLE && lp->verbose)
    printf("%% level %d INF\n", Level);

  if(failure == OPTIMAL) { /* there is a good solution */
    construct_solution(lp);

    /* because of reports of solution > upbo */
    /* check_solution(lp, upbo, lowbo); get too many hits ?? */

    debug_print(lp, "a solution was found");
    debug_print_solution(lp);

    /* if this solution is worse than the best sofar, this branch must die */

    /* if we can only have integer OF values, we might consider requiring to
       be at least 1 better than the best sofar, MB */

    if(lp->maximise)
      is_worse = lp->solution[0] <= lp->best_solution[0];
    else /* minimising! */
      is_worse = lp->solution[0] >= lp->best_solution[0];

    if(is_worse) {
      if(lp->verbose)
	printf("%% level %d OPT NOB value %g bound %g\n",
		Level, (double)lp->solution[0],
		(double)lp->best_solution[0]);
      debug_print(lp, "but it was worse than the best sofar, discarded");
      Level--;
      return(MILP_FAIL);
    }

    /* check if solution contains enough ints */
    notint = 0;
    if(lp->bb_rule == FIRST_NI) {
      for(notint = 0, i = lp->rows + 1;
	  i <= lp->sum && notint == 0;
	  i++) {
	if(lp->must_be_int[i] && !is_int(lp, i)) {
	  if(lowbo[i] == upbo[i]) { /* this var is already fixed */
	    printf("%% Warning: integer var %d is already fixed at %d, but has non-integer value %g\n",
		    i - lp->rows, (int)lowbo[i],
		    (double)lp->solution[i]);
	    printf("%% Perhaps the -e option should be used\n");
	  }
	  else
	    notint = i;
	}
      }
    }
    if(lp->bb_rule == RAND_NI) {
      int nr_not_int, select_not_int;
      nr_not_int = 0;

      for(i = lp->rows + 1; i <= lp->sum; i++)
	if(lp->must_be_int[i] && !is_int(lp, i))
	  nr_not_int++;

      if(nr_not_int == 0)
	notint = 0;
      else {
	select_not_int = (rand() % nr_not_int) + 1;
	i = lp->rows + 1;
	while(select_not_int > 0) {
	  if(lp->must_be_int[i] && !is_int(lp, i))
	    select_not_int--;
	  i++;
	}
	notint = i - 1;
      }
    }

    if(lp->verbose) {
      if(notint)
	printf("%% level %d OPT     value %g\n", Level,
		(double)lp->solution[0]);
      else
	printf("%% level %d OPT INT value %g\n", Level,
		(double)lp->solution[0]);
    }

    if(notint) { /* there is at least one value not yet int */
      /* set up two new problems */
      REAL   *new_upbo, *new_lowbo;
      REAL   new_bound;
      short  *new_lower,*new_basis;
      int    *new_bas;
      int     resone, restwo;

      /* allocate room for them */
      MALLOC(new_upbo,  lp->sum + 1);
      MALLOC(new_lowbo, lp->sum + 1);
      MALLOC(new_lower, lp->sum + 1);
      MALLOC(new_basis, lp->sum + 1);
      MALLOC(new_bas,   lp->rows + 1);
      memcpy(new_upbo,  upbo,      (lp->sum + 1)  * sizeof(REAL));
      memcpy(new_lowbo, lowbo,     (lp->sum + 1)  * sizeof(REAL));
      memcpy(new_lower, lp->lower, (lp->sum + 1)  * sizeof(short));
      memcpy(new_basis, lp->basis, (lp->sum + 1)  * sizeof(short));
      memcpy(new_bas,   lp->bas,   (lp->rows + 1) * sizeof(int));

      if(lp->names_used)
	debug_print(lp, "not enough ints. Selecting var %s, val: %g",
		    lp->col_name[notint - lp->rows],
		    (double)lp->solution[notint]);
      else
	debug_print(lp,
		    "not enough ints. Selecting Var [%d], val: %g",
		    notint, (double)lp->solution[notint]);
      debug_print(lp, "current bounds:\n");
      debug_print_bounds(lp, upbo, lowbo);

      if(lp->floor_first) {
	new_bound = ceil(lp->solution[notint]) - 1;

	/* this bound might conflict */
	if(new_bound < lowbo[notint]) {
	  debug_print(lp,
		      "New upper bound value %g conflicts with old lower bound %g\n",
		      (double)new_bound, (double)lowbo[notint]);
	  resone = MILP_FAIL;
	}
	else { /* bound feasible */
	  check_if_less(new_bound, upbo[notint], lp->solution[notint]);
	  new_upbo[notint] = new_bound;
	  debug_print(lp, "starting first subproblem with bounds:");
	  debug_print_bounds(lp, new_upbo, lowbo);
	  lp->eta_valid = FALSE;
	  resone = milpsolve(lp, new_upbo, lowbo, new_basis, new_lower,
			     new_bas, TRUE);
	  lp->eta_valid = FALSE;
	}
	new_bound += 1;
	if(new_bound > upbo[notint]) {
	  debug_print(lp,
		      "New lower bound value %g conflicts with old upper bound %g\n",
		      (double)new_bound, (double)upbo[notint]);
	  restwo = MILP_FAIL;
	}
	else { /* bound feasible */
	  check_if_less(lowbo[notint], new_bound, lp->solution[notint]);
	  new_lowbo[notint] = new_bound;
	  debug_print(lp, "starting second subproblem with bounds:");
	  debug_print_bounds(lp, upbo, new_lowbo);
	  lp->eta_valid = FALSE;
	  restwo = milpsolve(lp, upbo, new_lowbo, new_basis, new_lower,
			     new_bas, TRUE);
	  lp->eta_valid = FALSE;
	}
      }
      else { /* take ceiling first */
	new_bound = ceil(lp->solution[notint]);
	/* this bound might conflict */
	if(new_bound > upbo[notint]) {
	  debug_print(lp,
		      "New lower bound value %g conflicts with old upper bound %g\n",
		      (double)new_bound, (double)upbo[notint]);
	  resone = MILP_FAIL;
	}
	else { /* bound feasible */
	  check_if_less(lowbo[notint], new_bound, lp->solution[notint]);
	  new_lowbo[notint] = new_bound;
	  debug_print(lp, "starting first subproblem with bounds:");
	  debug_print_bounds(lp, upbo, new_lowbo);
	  lp->eta_valid = FALSE;
	  resone = milpsolve(lp, upbo, new_lowbo, new_basis, new_lower,
			     new_bas, TRUE);
	  lp->eta_valid = FALSE;
	}
	new_bound -= 1;
	if(new_bound < lowbo[notint]) {
	  debug_print(lp,
		      "New upper bound value %g conflicts with old lower bound %g\n",
		      (double)new_bound, (double)lowbo[notint]);
	  restwo = MILP_FAIL;
	}
	else { /* bound feasible */
	  check_if_less(new_bound, upbo[notint], lp->solution[notint]);
	  new_upbo[notint] = new_bound;
	  debug_print(lp, "starting second subproblem with bounds:");
	  debug_print_bounds(lp, new_upbo, lowbo);
	  lp->eta_valid = FALSE;
	  restwo = milpsolve(lp, new_upbo, lowbo, new_basis, new_lower,
			     new_bas, TRUE);
	  lp->eta_valid = FALSE;
	}
      }
      if(resone && restwo) /* both failed and must have been infeasible */
	failure = INFEASIBLE;
      else
	failure = OPTIMAL;

      free(new_upbo);
      free(new_lowbo);
      free(new_basis);
      free(new_lower);
      free(new_bas);
    }
    else { /* all required values are int */
      debug_print(lp, "--> valid solution found");

      if(lp->maximise)
	is_worse = lp->solution[0] < lp->best_solution[0];
      else
	is_worse = lp->solution[0] > lp->best_solution[0];

      if(!is_worse) { /* Current solution better */
	if(lp->debug || (lp->verbose && !lp->print_sol))
	  printf("%% *** new best solution: old: %g, new: %g ***\n",
		  (double)lp->best_solution[0], (double)lp->solution[0]);
	memcpy(lp->best_solution, lp->solution, (lp->sum + 1) * sizeof(REAL));
	calculate_duals(lp);

	if(lp->print_sol)
	  print_solution(lp);

	if(lp->break_at_int) {
	  if(lp->maximise && (lp->best_solution[0] > lp->break_value))
	    Break_bb = TRUE;

	  if(!lp->maximise && (lp->best_solution[0] < lp->break_value))
	    Break_bb = TRUE;
	}
      }
    }
  }

  Level--;

  /* failure can have the values OPTIMAL, UNBOUNDED and INFEASIBLE. */
  return(failure);
} /* milpsolve */


int solve(lprec *lp)
{
  int result, i;

  lp->total_iter  = 0;
  lp->max_level   = 1;
  lp->total_nodes = 0;

  if(isvalid(lp)) {
    if(lp->maximise && lp->obj_bound == lp->infinite)
      lp->best_solution[0] = -lp->infinite;
    else if(!lp->maximise && lp->obj_bound == -lp->infinite)
      lp->best_solution[0] = lp->infinite;
    else
      lp->best_solution[0] = lp->obj_bound;

    Level = 0;

    if(!lp->basis_valid) {
      for(i = 0; i <= lp->rows; i++) {
	lp->basis[i] = TRUE;
	lp->bas[i]   = i;
      }

      for(i = lp->rows + 1; i <= lp->sum; i++)
	lp->basis[i] = FALSE;

      for(i = 0; i <= lp->sum; i++)
	lp->lower[i] = TRUE;

      lp->basis_valid = TRUE;
    }

    lp->eta_valid = FALSE;
    Break_bb      = FALSE;
    result        = milpsolve(lp, lp->orig_upbo, lp->orig_lowbo, lp->basis,
			      lp->lower, lp->bas, FALSE);
    return(result);
  }

  /* if we get here, isvalid(lp) failed. I suggest we return FAILURE */
  printf("%% Error, the current LP seems to be invalid\n");
  return(FAILURE);
} /* solve */


/*
 * This routine saves the current basis of the given LP.
 */

	void
save_LP_basis (

lprec *			lp,		/* IN - LP to save basis for */
struct basis_save *	basp		/* OUT - saved basis info */
)
{
int		i;
int		j;
int		varnr;
int		rows;
int		sum;
double		theta;

	if (!lp->row_end_valid)
		set_row_end (lp);

	/* Re-initialize the various versions of things needed	*/
	/* to re-invert...					*/
	memcpy (lp->upbo,	lp->orig_upbo,	(lp->sum + 1) * sizeof (REAL));
	memcpy (lp->lowbo,	lp->orig_lowbo,	(lp->sum + 1) * sizeof (REAL));
	memcpy (lp->rh,		lp->orig_rh,	(lp->rows + 1) * sizeof (REAL));

	/* Process non-zero lower bounds... */
	for (i = 1; i <= lp->columns; i++) {
		varnr = lp->rows + i;
		theta = lp->lowbo[varnr];
		if (theta != 0.0) {
			if (lp->upbo[varnr] < lp->infinite) {
				lp->upbo[varnr] -= theta;
			}
			for (j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
				lp->rh[lp->mat[j].row_nr]
					-= theta * lp->mat[j].value;
			}
		}
	}

	/* First things first -- reinvert so that the eta	*/
	/* vectors are as simple as possible before we save...	*/
	invert (lp);

	lp -> eta_valid = TRUE;

	rows	= lp -> rows;
	sum	= lp -> sum;

	/* Allocate buffers to save state into... */
	CALLOC (basp -> bas, rows + 1);
	CALLOC (basp -> basis, sum + 1);
	CALLOC (basp -> lower, sum + 1);
	CALLOC (basp -> rhs, rows + 1);
	CALLOC (basp -> drow, sum + 1);
	CALLOC (basp -> prow, sum + 1);
	CALLOC (basp -> Pcol, rows + 1);

	/* Save the basis state info... */
	basp -> eta_size = lp -> eta_size;
	memcpy (basp -> bas, lp -> bas, (rows + 1) * sizeof (int));
	memcpy (basp -> basis, lp -> basis, (sum + 1) * sizeof (short));
	memcpy (basp -> lower, lp -> lower, (sum + 1) * sizeof (short));
	memcpy (basp -> rhs, lp -> rhs, (rows + 1) * sizeof (REAL));
}

/*
 * Destroy the saved basis info...
 */

	void
destroy_LP_basis (

struct basis_save *	basp		/* IN - basis info to free up */
)
{
	free ((char *) (basp -> Pcol));
	free ((char *) (basp -> prow));
	free ((char *) (basp -> drow));
	free ((char *) (basp -> rhs));
	free ((char *) (basp -> lower));
	free ((char *) (basp -> basis));
	free ((char *) (basp -> bas));
}


/*
 * This routine performs a quick "test run" of branching the given
 * variable in the given direction.  We start from the current basis
 * (which has been saved in basp, and is assumed optimal), and do at
 * most 50 or so iterations -- but NEVER reinverting, which would
 * clobber the saved eta vectors...  This means we can restore the
 * eta just by popping back to the initial eta size.
 */

	REAL
try_branch (

lprec *			lp,	/* IN - LP to test branching for */
int			var,	/* IN - variable to branch */
int			dir,	/* IN - direction to branch in, 0 or 1 */
REAL *			x,	/* OUT - LP solution found */
REAL			ival,	/* IN - value to yield if infeasible */
struct basis_save *	basp	/* IN - basis to restore when done */
)
{
int		i;
int		j;
int		varnr;
REAL		f;
REAL		theta;
short		primal;
REAL *		drow;
REAL *		prow;
REAL *		Pcol;
REAL		save_lb;
REAL		save_ub;
short		minit;
int		colnr;
int		row_nr;

	lp -> total_iter	= 0;
	lp -> max_level		= 1;
	lp -> total_nodes	= 0;

	if ((var < 1) ||
	    (var > lp -> columns) ||
	    !isvalid (lp) ||
	    (lp -> scaling_used) ||
	    !(lp -> basis_valid) ||
	    !(lp -> eta_valid) ||
	    !(lp -> basis [lp->rows + var])) {
		fprintf (stderr, "Bad call to try_branch...\n");
		abort ();
	}

	Level = 1;

	drow	= basp -> drow;
	prow	= basp -> prow;
	Pcol	= basp -> Pcol;

	lp -> iter = 0;
	minit = FALSE;
	Status = RUNNING;
	DoInvert = FALSE;
	Doiter = FALSE;

	/* We assume the caller starts us from an optimal basis, from	*/
	/* which we branch (up or down), creating an infeasibility.	*/
	/* Thus after branching we always use the Dual Simplex...	*/

	varnr = lp -> rows + var;
	save_lb = lp -> lowbo [varnr];
	save_ub = lp -> upbo [varnr];
	if (dir == 0) {
		/* Branching down -- fix upper bound and continue	*/
		/* optimizing from the same tableaux...			*/
		lp -> upbo [varnr] = 0.0;
	}
	else {
		/* Branching up -- adjust rhs.  The branching variable	*/
		/* MUST be basis!  (We verified this above.)  Therefore	*/
		/* there is only a single RHS value that we need to	*/
		/* adjust because this variables column is all zeros	*/
		/* except for a one in the row in which it is basic.	*/
		lp -> lowbo [varnr] = 1.0;
		lp -> upbo [varnr] = 0.0;
		for (i = 1; ; i++) {
			if (i > lp -> rows) {
				fprintf (stderr, "try_branch: Bug 1.\n");
				abort ();
			}
			if (lp -> bas [i] == varnr) {
				lp -> rhs [i] -= 1.0;
				break;
			}
		}
		/* Fix up objective function value also! */
		for (j = lp -> col_end [var - 1]; j < lp -> col_end [var]; j++) {
			if (lp -> mat [j].row_nr == 0) {
				lp -> rhs [0] -= lp -> mat [j].value;
				break;
			}
		}
	}

	primal = FALSE;

	/* Get most negative INITIAL objective coefficient into	*/
	/* Extrad...						*/
	drow [0] = 1;
	for (i = 1; i <= lp -> rows; i++) {
		drow [i] = 0;
	}
	Extrad = 0;
	for (i = 1; i <= lp -> columns; i++) {
		varnr = lp -> rows + i;
		drow [varnr] = 0;
		for (j = lp -> col_end [i - 1]; j < lp -> col_end [i]; j++) {
			if (drow [lp -> mat[j].row_nr] != 0) {
				drow [varnr] +=
					drow [lp -> mat [j].row_nr]
					* lp -> mat [j].value;
			}
		}
		if (drow [varnr] < Extrad) {
			Extrad = drow [varnr];
		}
	}
	if (lp -> trace) {
		printf ("%% Extrad = %f\n", Extrad);
	}

	row_nr = 0;
	colnr = 0;

	minit = FALSE;

	while (Status == RUNNING) {
		Doiter = FALSE;
		DoInvert = FALSE;

		if (!minit) {
			row_nr = rowdual (lp);
		}
		if (row_nr > 0 ) {
			colnr = coldual (lp, row_nr, minit, prow, drow);
			if (colnr > 0) {
				setpivcol (lp, colnr, Pcol);
				/* getting div by zero here ... MB */
				if (Pcol [row_nr] == 0) {
					printf ("%% An attempt was made to divide by zero (Pcol[%d])\n",
						 row_nr);
					printf ("%% This indicates numerical instability\n");
					Doiter = FALSE;
					Status = STOP_AT_INVERT;
				}
				else {
					condensecol (lp, row_nr, Pcol);
					f = lp -> rhs [row_nr]
						- lp -> upbo [lp -> bas [row_nr]];
					if (f > 0) {
						theta = f / (REAL) Pcol [row_nr];
						if (theta <= lp -> upbo [colnr] + lp -> epsb) {
							lp -> lower [lp -> bas [row_nr]]
								= !lp -> lower [lp -> bas [row_nr]];
						}
					}
					else { /* f <= 0 */
						theta = lp -> rhs [row_nr] / (REAL) Pcol [row_nr];
					}
				}
			}
			else {
				Status = INFEASIBLE;
			}
		}
		else {
			/* Assume optimal achieved... */
			Doiter   = FALSE;
			DoInvert = TRUE;
		}
		if (Doiter) {
			iteration (lp,
				   row_nr,
				   colnr,
				   &theta,
				   lp -> upbo [colnr],
				   &minit,
				   &(lp -> lower [colnr]),
				   primal,
				   Pcol);
		}
		if (lp -> num_inv >= lp -> max_num_inv) {
			DoInvert = TRUE;
		}
		if (DoInvert) {
			/* We DO NOT invert!  That would mess up the	*/
			/* eta vectors that we want to restore back to!	*/
			/* Instead we just stop here and we are done!	*/
			Status = STOP_AT_INVERT;
		}
	}

	lp -> total_iter += lp -> iter;

	construct_solution (lp);
	memcpy (x,
		&(lp -> solution [lp -> rows + 1]),
		lp -> columns * sizeof (REAL));

	/* Restore the LP state back to the saved basis... */
	varnr = lp -> rows + var;
	lp -> lowbo [varnr] = save_lb;
	lp -> upbo [varnr] = save_ub;
	lp -> eta_size = basp -> eta_size;

	memcpy (lp -> bas, basp -> bas, (lp -> rows + 1) * sizeof (int));
	memcpy (lp -> basis, basp -> basis, (lp -> sum + 1) * sizeof (short));
	memcpy (lp -> lower, basp -> lower, (lp -> sum + 1) * sizeof (short));
	memcpy (lp -> rhs, basp -> rhs, (lp -> rows + 1) * sizeof (REAL));

	if (Status == INFEASIBLE) {
		return (ival);
	}

	return (lp -> solution [0]);
}

int lag_solve(lprec *lp, REAL start_bound, int num_iter, short verbose)
{
  int i, j, result, citer;
  short status, OrigFeas, AnyFeas, same_basis;
  REAL *OrigObj, *ModObj, *SubGrad, *BestFeasSol;
  REAL Zub, Zlb, Ztmp, pie;
  REAL rhsmod, Step, SqrsumSubGrad;
  int   *old_bas;
  short *old_lower;

  /* allocate mem */
  MALLOC(OrigObj, lp->columns + 1);
  CALLOC(ModObj, lp->columns + 1);
  CALLOC(SubGrad, lp->nr_lagrange);
  CALLOC(BestFeasSol, lp->sum + 1);
  MALLOCCPY(old_bas, lp->bas, lp->rows + 1);
  MALLOCCPY(old_lower, lp->lower, lp->sum + 1);

  get_row(lp, 0, OrigObj);

  pie = 2;

  if(lp->maximise) {
    Zub = DEF_INFINITE;
    Zlb = start_bound;
  }
  else {
    Zlb = -DEF_INFINITE;
    Zub = start_bound;
  }
  status   = RUNNING;
  Step     = 1;
  OrigFeas = FALSE;
  AnyFeas  = FALSE;
  citer    = 0;

  for(i = 0 ; i < lp->nr_lagrange; i++)
    lp->lambda[i] = 0;

  while(status == RUNNING) {
    citer++;

    for(i = 1; i <= lp->columns; i++) {
      ModObj[i] = OrigObj[i];
      for(j = 0; j < lp->nr_lagrange; j++) {
	if(lp->maximise)
	  ModObj[i] -= lp->lambda[j] * lp->lag_row[j][i];
	else
	  ModObj[i] += lp->lambda[j] * lp->lag_row[j][i];
      }
    }
    for(i = 1; i <= lp->columns; i++) {
      set_mat(lp, 0, i, ModObj[i]);
    }
    rhsmod = 0;
    for(i = 0; i < lp->nr_lagrange; i++)
      if(lp->maximise)
	rhsmod += lp->lambda[i] * lp->lag_rhs[i];
      else
	rhsmod -= lp->lambda[i] * lp->lag_rhs[i];

    if(verbose) {
      printf("%% Zub: %10g Zlb: %10g Step: %10g pie: %10g Feas %d\n",
	      (double)Zub, (double)Zlb, (double)Step, (double)pie, OrigFeas);
      for(i = 0; i < lp->nr_lagrange; i++)
	printf("%% %3d SubGrad %10g lambda %10g\n", i,
		(double)SubGrad[i], (double)lp->lambda[i]);
    }

    if(verbose && lp->sum < 20)
      print_lp(lp);

    result = solve(lp);

    if(verbose && lp->sum < 20) {
      print_solution(lp);
    }

    same_basis = TRUE;
    i = 1;
    while(same_basis && i < lp->rows) {
      same_basis = (old_bas[i] == lp->bas[i]);
      i++;
    }
    i = 1;
    while(same_basis && i < lp->sum) {
      same_basis=(old_lower[i] == lp->lower[i]);
      i++;
    }
    if(!same_basis) {
      memcpy(old_lower, lp->lower, (lp->sum+1) * sizeof(short));
      memcpy(old_bas, lp->bas, (lp->rows+1) * sizeof(int));
      pie *= 0.95;
    }

    if(verbose)
      printf("%% result: %d  same basis: %d\n", result, same_basis);

    if(result == UNBOUNDED) {
      for(i = 1; i <= lp->columns; i++)
	printf("%g ", (double)ModObj[i]);
      exit(EXIT_FAILURE);
    }

    if(result == FAILURE)
      status = FAILURE;

    if(result == INFEASIBLE)
      status = INFEASIBLE;

    SqrsumSubGrad = 0;
    for(i = 0; i < lp->nr_lagrange; i++) {
      SubGrad[i]= -lp->lag_rhs[i];
      for(j = 1; j <= lp->columns; j++)
	SubGrad[i] += lp->best_solution[lp->rows + j] * lp->lag_row[i][j];
      SqrsumSubGrad += SubGrad[i] * SubGrad[i];
    }

    OrigFeas = TRUE;
    for(i = 0; i < lp->nr_lagrange; i++)
      if(lp->lag_con_type[i]) {
	if(my_abs(SubGrad[i]) > lp->epsb)
	  OrigFeas = FALSE;
      }
      else if(SubGrad[i] > lp->epsb)
	OrigFeas = FALSE;

    if(OrigFeas) {
      AnyFeas = TRUE;
      Ztmp = 0;
      for(i = 1; i <= lp->columns; i++)
	Ztmp += lp->best_solution[lp->rows + i] * OrigObj[i];
      if((lp->maximise) && (Ztmp > Zlb)) {
	Zlb = Ztmp;
	for(i = 1; i <= lp->sum; i++)
	  BestFeasSol[i] = lp->best_solution[i];
	BestFeasSol[0] = Zlb;
	if(verbose)
	  printf("%% Best feasible solution: %g\n", (double)Zlb);
      }
      else if(Ztmp < Zub) {
	Zub = Ztmp;
	for(i = 1; i <= lp->sum; i++)
	  BestFeasSol[i] = lp->best_solution[i];
	BestFeasSol[0] = Zub;
	if(verbose)
	  printf("%% Best feasible solution: %g\n", (double)Zub);
      }
        }

    if(lp->maximise)
      Zub = my_min(Zub, rhsmod + lp->best_solution[0]);
    else
      Zlb = my_max(Zlb, rhsmod + lp->best_solution[0]);

    if(my_abs(Zub-Zlb)<0.001) {
      status = OPTIMAL;
    }
    Step = pie * ((1.05*Zub) - Zlb) / SqrsumSubGrad;

    for(i = 0; i < lp->nr_lagrange; i++) {
      lp->lambda[i] += Step * SubGrad[i];
      if(!lp->lag_con_type[i] && lp->lambda[i] < 0)
	lp->lambda[i] = 0;
    }

    if(citer == num_iter && status==RUNNING) {
      if(AnyFeas)
	status = FEAS_FOUND;
      else
	status = NO_FEAS_FOUND;
    }
  }

  for(i = 0; i <= lp->sum; i++)
    lp->best_solution[i] = BestFeasSol[i];

  for(i = 1; i <= lp->columns; i++)
    set_mat(lp, 0, i, OrigObj[i]);

  if(lp->maximise)
    lp->lag_bound = Zub;
  else
    lp->lag_bound = Zlb;
  free(BestFeasSol);
  free(SubGrad);
  free(OrigObj);
  free(ModObj);
  free(old_bas);
  free(old_lower);

  return(status);
}
