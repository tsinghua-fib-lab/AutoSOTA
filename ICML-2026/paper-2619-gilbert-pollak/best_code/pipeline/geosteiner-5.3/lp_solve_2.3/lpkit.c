/* 11/18/2022: warme: Fix -Wall issues. */

#include "lpkit.h"
#include "lpglob.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define HASHSIZE 10007

/* Globals */
int     Rows;
int     Columns;
int     Sum;
int     Non_zeros;
int     Level;

REAL	Trej;

short   Maximise;
REAL    Extrad;

int     Warn_count; /* used in CHECK version of rounding macro */

void error(char *format, ...)
{
  va_list ap;

  va_start(ap, format);
  vfprintf(stderr, format, ap);
  fputc('\n', stderr);
  va_end(ap);

  exit(EXIT_FAILURE);
}

lprec *make_lp(int rows, int columns)
{
  lprec *newlp;
  int i, sum;  

  if(rows < 0 || columns < 0)
    error("rows < 0 or columns < 0");

  sum = rows + columns;

  CALLOC(newlp, 1);

  strcpy(newlp->lp_name, "unnamed");

  newlp->verbose = FALSE;
  newlp->print_duals = FALSE;
  newlp->print_sol = FALSE;
  newlp->debug = FALSE;
  newlp->print_at_invert = FALSE;
  newlp->trace = FALSE;

  newlp->rows = rows;
  newlp->columns = columns;
  newlp->sum = sum;
  newlp->rows_alloc = rows;
  newlp->columns_alloc = columns;
  newlp->sum_alloc = sum;
  newlp->names_used = FALSE;

  newlp->obj_bound = DEF_INFINITE;
  newlp->infinite = DEF_INFINITE;
  newlp->epsilon = DEF_EPSILON;
  newlp->epsb = DEF_EPSB;
  newlp->epsd = DEF_EPSD;
  newlp->epsel = DEF_EPSEL;
  newlp->non_zeros = 0;
  newlp->mat_alloc = 1;
  CALLOC(newlp->mat, newlp->mat_alloc);
  CALLOC(newlp->col_no, newlp->mat_alloc + 1);
  CALLOC(newlp->col_end, columns + 1);
  CALLOC(newlp->row_end, rows + 1);
  newlp->row_end_valid = FALSE;
  CALLOC(newlp->orig_rh, rows + 1);
  CALLOC(newlp->rh, rows + 1);
  CALLOC(newlp->rhs, rows + 1);

  CALLOC(newlp->must_be_int, sum + 1);
  for(i = 0; i <= sum; i++)
    newlp->must_be_int[i] = FALSE;

  CALLOC(newlp->orig_upbo, sum + 1);
  for(i = 0; i <= sum; i++)
    newlp->orig_upbo[i] = newlp->infinite;

  CALLOC(newlp->upbo, sum + 1);
  CALLOC(newlp->orig_lowbo, sum + 1);
  CALLOC(newlp->lowbo, sum + 1);

  newlp->basis_valid = TRUE;
  CALLOC(newlp->bas, rows+1);
  CALLOC(newlp->basis, sum + 1);
  CALLOC(newlp->lower, sum + 1);
  newlp->colsign = NULL;

  for(i = 0; i <= rows; i++) {
    newlp->bas[i] = i;
    newlp->basis[i] = TRUE;
  }

  for(i = rows + 1; i <= sum; i++)
    newlp->basis[i] = FALSE;


  for(i = 0 ; i <= sum; i++)
    newlp->lower[i] = TRUE;
 
  newlp->eta_valid = TRUE;
  newlp->eta_size = 0;
  newlp->eta_alloc = INITIAL_MAT_SIZE;
  newlp->max_num_inv = DEFNUMINV;

  newlp->nr_lagrange = 0;

  CALLOC(newlp->eta_value, newlp->eta_alloc);
  CALLOC(newlp->eta_row_nr, newlp->eta_alloc);

  /* +1 reported by Christian Rank */
  CALLOC(newlp->eta_col_end, newlp->rows_alloc + newlp->max_num_inv + 1);

  newlp->bb_rule = FIRST_NI;
  newlp->break_at_int = FALSE;
  newlp->break_value = 0;

  newlp->iter = 0;
  newlp->total_iter = 0;

  CALLOC(newlp->solution, sum + 1);
  CALLOC(newlp->best_solution, sum + 1);
  CALLOC(newlp->duals, rows + 1);

  newlp->maximise = FALSE;
  newlp->floor_first = TRUE;

  newlp->scaling_used = FALSE;
  newlp->columns_scaled = FALSE;

  CALLOC(newlp->ch_sign, rows + 1);

  for(i = 0; i <= rows; i++)
    newlp->ch_sign[i] = FALSE;

  newlp->valid = FALSE; 

  /* create two hash tables for names */
  newlp->rowname_hashtab = create_hash_table(HASHSIZE);
  newlp->colname_hashtab = create_hash_table(HASHSIZE);

  return(newlp);
}

void delete_lp(lprec *lp)
{
  int i; 

  if(lp->names_used) {
    free(lp->row_name);
    free(lp->col_name);
  }

  free(lp->mat);
  free(lp->col_no);
  free(lp->col_end);
  free(lp->row_end);
  free(lp->orig_rh);
  free(lp->rh);
  free(lp->rhs);
  free(lp->must_be_int);
  free(lp->orig_upbo);
  free(lp->orig_lowbo);
  free(lp->upbo);
  free(lp->lowbo);
  free(lp->bas);
  free(lp->basis);
  free(lp->lower);
  if (lp->colsign)
    free(lp->colsign);
  free(lp->eta_value);
  free(lp->eta_row_nr);
  free(lp->eta_col_end);
  free(lp->solution);
  free(lp->best_solution);
  free(lp->duals);
  free(lp->ch_sign);
  if(lp->scaling_used)
    free(lp->scale);
  if(lp->nr_lagrange > 0) {
    free(lp->lag_rhs);
    free(lp->lambda);
    free(lp->lag_con_type);
    for(i = 0; i < lp->nr_lagrange; i++)
      free(lp->lag_row[i]);
    free(lp->lag_row);
  }

  free_hash_table(lp->rowname_hashtab);
  free_hash_table(lp->colname_hashtab);

  free(lp);

}  

lprec *copy_lp(lprec *lp)
{
  lprec *newlp;
  int i, rowsplus, colsplus, sumplus;
  
  rowsplus = lp->rows_alloc + 1;
  colsplus = lp->columns_alloc + 1;
  sumplus  = lp->sum_alloc + 1;

  MALLOCCPY(newlp, lp, 1); /* copy all non pointers */

  if(newlp->names_used) {
    MALLOCCPY(newlp->col_name, lp->col_name, colsplus);
    MALLOCCPY(newlp->row_name, lp->row_name, rowsplus);
  }

  newlp->rowname_hashtab = copy_hash_table(lp->rowname_hashtab);
  newlp->colname_hashtab = copy_hash_table(lp->colname_hashtab);

  MALLOCCPY(newlp->mat, lp->mat, newlp->mat_alloc);
  MALLOCCPY(newlp->col_end, lp->col_end, colsplus);
  MALLOCCPY(newlp->col_no, lp->col_no, newlp->mat_alloc + 1);
  MALLOCCPY(newlp->row_end, lp->row_end, rowsplus);
  MALLOCCPY(newlp->orig_rh, lp->orig_rh, rowsplus);
  MALLOCCPY(newlp->rh, lp->rh, rowsplus);
  MALLOCCPY(newlp->rhs, lp->rhs, rowsplus);
  MALLOCCPY(newlp->must_be_int, lp->must_be_int, sumplus);
  MALLOCCPY(newlp->orig_upbo, lp->orig_upbo, sumplus);
  MALLOCCPY(newlp->orig_lowbo, lp->orig_lowbo, sumplus);
  MALLOCCPY(newlp->upbo, lp->upbo, sumplus);
  MALLOCCPY(newlp->lowbo, lp->lowbo, sumplus);
  MALLOCCPY(newlp->bas, lp->bas, rowsplus);
  MALLOCCPY(newlp->basis, lp->basis, sumplus);
  MALLOCCPY(newlp->lower, lp->lower, sumplus);
  newlp->colsign = NULL;
  MALLOCCPY(newlp->eta_value, lp->eta_value, lp->eta_alloc);
  MALLOCCPY(newlp->eta_row_nr, lp->eta_row_nr, lp->eta_alloc);
  MALLOCCPY(newlp->eta_col_end, lp->eta_col_end,
	    lp->rows_alloc + lp->max_num_inv + 1);
  MALLOCCPY(newlp->solution, lp->solution, sumplus);
  MALLOCCPY(newlp->best_solution, lp->best_solution, sumplus);
  MALLOCCPY(newlp->duals, lp->duals, rowsplus);
  MALLOCCPY(newlp->ch_sign, lp->ch_sign, rowsplus);

  if(newlp->scaling_used)
    MALLOCCPY(newlp->scale, lp->scale, sumplus);

  if(newlp->nr_lagrange > 0) {
    MALLOCCPY(newlp->lag_rhs, lp->lag_rhs, newlp->nr_lagrange);
    MALLOCCPY(newlp->lambda, lp->lambda, newlp->nr_lagrange);
    MALLOCCPY(newlp->lag_con_type, lp->lag_con_type, newlp->nr_lagrange);
    MALLOC(newlp->lag_row, newlp->nr_lagrange);
    for(i = 0; i < newlp->nr_lagrange; i++)
      MALLOCCPY(newlp->lag_row[i], lp->lag_row[i], colsplus);
  }
  return(newlp);
}

void inc_mat_space(lprec *lp, int maxextra)
{
   if(lp->non_zeros + maxextra >= lp->mat_alloc) {
     /* let's allocate at least INITIAL_MAT_SIZE  entries */
     if(lp->mat_alloc < INITIAL_MAT_SIZE)
       lp->mat_alloc = INITIAL_MAT_SIZE;
     
     /* increase the size by 50% each time it becomes too small */
     while(lp->non_zeros + maxextra >= lp->mat_alloc)
       lp->mat_alloc *= 1.5;

     REALLOC(lp->mat, lp->mat_alloc);
     REALLOC(lp->col_no, lp->mat_alloc + 1);
   }
}
 
void inc_row_space(lprec *lp)
{
  if(lp->rows > lp->rows_alloc) {
    lp->rows_alloc = lp->rows+10;
    lp->sum_alloc  = lp->rows_alloc + lp->columns_alloc;
    REALLOC(lp->orig_rh, lp->rows_alloc + 1);
    REALLOC(lp->rh, lp->rows_alloc + 1);
    REALLOC(lp->rhs, lp->rows_alloc + 1);
    REALLOC(lp->orig_upbo, lp->sum_alloc + 1);
    REALLOC(lp->upbo, lp->sum_alloc + 1);
    REALLOC(lp->orig_lowbo, lp->sum_alloc + 1);
    REALLOC(lp->lowbo, lp->sum_alloc + 1);
    REALLOC(lp->solution, lp->sum_alloc + 1);
    REALLOC(lp->best_solution, lp->sum_alloc + 1);
    REALLOC(lp->row_end, lp->rows_alloc + 1);
    REALLOC(lp->basis, lp->sum_alloc + 1);
    REALLOC(lp->lower, lp->sum_alloc + 1);
    REALLOC(lp->must_be_int, lp->sum_alloc + 1);
    REALLOC(lp->bas, lp->rows_alloc + 1);
    REALLOC(lp->duals, lp->rows_alloc + 1);
    REALLOC(lp->ch_sign, lp->rows_alloc + 1);
    REALLOC(lp->eta_col_end, lp->rows_alloc + lp->max_num_inv + 1);
    if(lp->names_used)
      REALLOC(lp->row_name, lp->rows_alloc + 1);
    if(lp->scaling_used)
      REALLOC(lp->scale, lp->sum_alloc + 1);
  }
}

void inc_col_space(lprec *lp)
{
  if(lp->columns >= lp->columns_alloc) {
    lp->columns_alloc = lp->columns + 10;
    lp->sum_alloc = lp->rows_alloc + lp->columns_alloc;
    REALLOC(lp->must_be_int, lp->sum_alloc + 1);
    REALLOC(lp->orig_upbo, lp->sum_alloc + 1);
    REALLOC(lp->upbo, lp->sum_alloc + 1);
    REALLOC(lp->orig_lowbo, lp->sum_alloc + 1);
    REALLOC(lp->lowbo, lp->sum_alloc + 1);
    REALLOC(lp->solution, lp->sum_alloc + 1);
    REALLOC(lp->best_solution, lp->sum_alloc + 1);
    REALLOC(lp->basis, lp->sum_alloc + 1);
    REALLOC(lp->lower, lp->sum_alloc + 1);
    if(lp->names_used)
      REALLOC(lp->col_name, lp->columns_alloc + 1);
    if(lp->scaling_used)
      REALLOC(lp->scale, lp->sum_alloc + 1);
    REALLOC(lp->col_end, lp->columns_alloc + 1);
  }
}

void set_mat(lprec *lp, int Row, int Column, REAL Value)
{
  int elmnr, lastelm, i;

  /* This function is very inefficient if used to add new matrix entries in
     other places than at the end of the matrix. OK for replacing existing
     non-zero values */

  if(Row > lp->rows || Row < 0)
    error("Row out of range");
  if(Column > lp->columns || Column < 1)
    error("Column out of range");

  /* scaling is performed twice? MB */
  if(lp->scaling_used)
    Value *= lp->scale[Row] * lp->scale[lp->rows + Column];
  
  if (lp->basis[Column] == TRUE && Row > 0)
    lp->basis_valid = FALSE;
  lp->eta_valid = FALSE;

  /* find out if we already have such an entry */
  elmnr = lp->col_end[Column - 1];
  while((elmnr < lp->col_end[Column]) && (lp->mat[elmnr].row_nr != Row))
    elmnr++;
  
  if((elmnr != lp->col_end[Column]) && (lp->mat[elmnr].row_nr == Row)) {
    /* there is an existing entry */
    if(my_abs(Value) > lp->epsilon) { /* we replace it by something non-zero */
      if (lp->scaling_used) {
	if(lp->ch_sign[Row])
	  lp->mat[elmnr].value = -Value * lp->scale[Row] * lp->scale[Column];
	else
	  lp->mat[elmnr].value = Value * lp->scale[Row] * lp->scale[Column];
      }
      else { /* no scaling */
	if(lp->ch_sign[Row])
	  lp->mat[elmnr].value = -Value;
	else
	  lp->mat[elmnr].value = Value;
      }
    }
    else { /* setting existing non-zero entry to zero. Remove the entry */
      /* this might remove an entire column, or leave just a bound. No
	 nice solution for that yet */
      
      /* Shift the matrix */
      lastelm = lp->non_zeros; 
      for(i = elmnr; i < lastelm ; i++)
	lp->mat[i] = lp->mat[i + 1];
      for(i = Column; i <= lp->columns; i++)
	lp->col_end[i]--;
      
      lp->non_zeros--;
    }
  }
  else if(my_abs(Value) > lp->epsilon) {
    /* no existing entry. make new one only if not nearly zero */
    /* check if more space is needed for matrix */
    inc_mat_space(lp, 1);
    
    /* Shift the matrix */
    lastelm = lp->non_zeros; 
    for(i = lastelm; i > elmnr ; i--)
      lp->mat[i] = lp->mat[i - 1];
    for(i = Column; i <= lp->columns; i++)
      lp->col_end[i]++;
    
    /* Set new element */
    lp->mat[elmnr].row_nr = Row;
    
    if (lp->scaling_used) {
      if(lp->ch_sign[Row])
	lp->mat[elmnr].value = -Value * lp->scale[Row] * lp->scale[Column];
      else
	lp->mat[elmnr].value = Value * lp->scale[Row] * lp->scale[Column];
    }
    else /* no scaling */
      {
	if(lp->ch_sign[Row])
	  lp->mat[elmnr].value = -Value;
	else
	  lp->mat[elmnr].value = Value;
      }
    
    lp->row_end_valid = FALSE;
    
    lp->non_zeros++;
  }      
}

void set_obj_fn(lprec *lp, REAL *row)
{
  int i;
  for(i = 1; i <= lp->columns; i++)
    set_mat(lp, 0, i, row[i]);
}

void str_set_obj_fn(lprec *lp, char *row)
{
  int  i;
  REAL *arow;
  char *p, *newp;
  CALLOC(arow, lp->columns + 1);
  p = row;
  for(i = 1; i <= lp->columns; i++) {
    arow[i] = (REAL) strtod(p, &newp);
    if(p == newp)
      error("Bad string in str_set_obj_fn");
    else
      p = newp; 
  }
  set_obj_fn(lp, arow);
  free(arow);
}


void add_constraint(lprec *lp, REAL *row, short constr_type, REAL rh)
{
  matrec *newmat;
  int  i, j;
  int  elmnr;
  int  stcol;
  int  *addtoo;

  MALLOC(addtoo, lp->columns + 1);

    for(i = 1; i <= lp->columns; i++)
      if(row[i] != 0) {
	addtoo[i] = TRUE;
	lp->non_zeros++;
      }
      else
	addtoo[i] = FALSE;

  MALLOC(newmat, lp->non_zeros);
  inc_mat_space(lp, 0);
  lp->rows++;
  lp->sum++;
  inc_row_space(lp);

  if(lp->scaling_used) {
    /* shift scale */
    for(i = lp->sum; i > lp->rows; i--)
      lp->scale[i] = lp->scale[i - 1];
    lp->scale[lp->rows] = 1;
  }

  if(lp->names_used)
    sprintf(lp->row_name[lp->rows], "r_%d", lp->rows);

  if(lp->scaling_used && lp->columns_scaled)
    for(i = 1; i <= lp->columns; i++)
      row[i] *= lp->scale[lp->rows + i];
     
  if(constr_type == REL_GE)
    lp->ch_sign[lp->rows] = TRUE;
  else
    lp->ch_sign[lp->rows] = FALSE;

  elmnr = 0;
  stcol = 0;
  for(i = 1; i <= lp->columns; i++) {
    for(j = stcol; j < lp->col_end[i]; j++) {  
      newmat[elmnr].row_nr = lp->mat[j].row_nr;
      newmat[elmnr].value = lp->mat[j].value;
      elmnr++;
    }
    if(addtoo[i]) {
      if(lp->ch_sign[lp->rows])
	newmat[elmnr].value = -row[i];
      else
	newmat[elmnr].value = row[i];
      newmat[elmnr].row_nr = lp->rows;
      elmnr++;
    }
    stcol = lp->col_end[i];
    lp->col_end[i] = elmnr;
  }    
  
  memcpy(lp->mat, newmat, lp->non_zeros * sizeof(matrec));
 
  free(newmat);
  free(addtoo);

  for(i = lp->sum; i > lp->rows; i--) {
    lp->orig_upbo[i]   = lp->orig_upbo[i - 1];
    lp->orig_lowbo[i]  = lp->orig_lowbo[i - 1];
    lp->basis[i]       = lp->basis[i - 1];
    lp->lower[i]       = lp->lower[i - 1];
    lp->must_be_int[i] = lp->must_be_int[i - 1];
  }

  /* changed from i <= lp->rows to i < lp->rows, MB */
  for(i = 1 ; i < lp->rows; i++)
    if(lp->bas[i] >= lp->rows)
      lp->bas[i]++;

  if(constr_type == REL_LE || constr_type == REL_GE) {
    lp->orig_upbo[lp->rows] = lp->infinite;
  }
  else if(constr_type == REL_EQ) {
    lp->orig_upbo[lp->rows] = 0;
  }
  else {
    fprintf(stderr, "Wrong constraint type\n");
    exit(EXIT_FAILURE);
  }

  lp->orig_lowbo[lp->rows] = 0;

  if(constr_type == REL_GE && rh != 0)
    lp->orig_rh[lp->rows] = -rh;
  else
    lp->orig_rh[lp->rows] = rh;  

  lp->row_end_valid = FALSE;
 
  lp->bas[lp->rows] = lp->rows;
  lp->basis[lp->rows] = TRUE;
  lp->lower[lp->rows] = TRUE;   
  lp->eta_valid = FALSE;
}

void str_add_constraint(lprec *lp,
			char *row_string,
			short constr_type,
			REAL rh)
{
  int  i;
  REAL *aRow;
  char *p, *newp;
  CALLOC(aRow, lp->columns + 1);
  p = row_string;
 
  for(i = 1; i <= lp->columns; i++) {
    aRow[i] = (REAL) strtod(p, &newp);
    if(p == newp)
      error("Bad string in str_add_constr");
    else
      p = newp; 
  }
  add_constraint(lp, aRow, constr_type, rh);
  free(aRow);
}

void add_rows(lprec *lp,
	      int ccnt,		/* # of new columns being added */
	      int rcnt,		/* # of new rows being added */
	      REAL *rh,         /* vector of rcnt right-hand-sides */
	      short *ctype,	/* vector of rcnt constraint types */
	      int *rmatbeg,	/* index of new row i's non-zeros */
	      int *rmatind,	/* column # of non-zero entries */
	      REAL *rmatval)	/* non-zero entries */
{
  int i, j, nzcnt;
  int *count;
  matrec **ptr1, **ptr2, *tmat;
  matrec *mp, *src, *dst, *end, *save;

  /* Handle new columns... */
  if(ccnt < 0 || rcnt < 0)
    error("Too few rows or columns.");

  if(rmatbeg[0] != 0)
    error("Invalid 'rmatbeg' array.");
  for(i = 0; i < rcnt; i++)
    if(rmatbeg[i+1] <= rmatbeg[i])
      error("Invalid 'rmatbeg' array.");
  nzcnt = rmatbeg[rcnt];

  for(i = 0; i < nzcnt; i++)
    if(rmatind[i] < 0 || rmatind[i] >= lp->columns + ccnt)
      error("Invalid 'rmatind' array.");

  lp->non_zeros += nzcnt;
  inc_mat_space(lp, 0);

  if(ccnt > 0)
    {
      lp->columns += ccnt;
      inc_col_space(lp);
      for(i = lp->sum + 1 - ccnt; i <= lp->sum; i++)
	{
	  lp->orig_lowbo[i] = 0;
	  lp->orig_upbo[i] = lp->infinite;
	}
    }

  if(rcnt > 0)
    {
      /* Handle new rows... */
      lp->rows += rcnt;
      lp->sum += rcnt;
      inc_row_space(lp);

      if(lp->scaling_used)
        {
	  /* shift scale */
	  for(i = lp->sum; i > lp->rows; i--)
	    lp->scale[i]=lp->scale[i-rcnt];
	  for(; i > lp->rows - rcnt; i--)
	    lp->scale[i] = 1;
	  if(lp->columns_scaled)
	    for(i = 0; i < nzcnt; i++)
	      rmatval[i] *= lp->scale[lp->rows + 1 + rmatind[i]];
	}

      if(lp->names_used)
	for(i = lp->rows + 1 - rcnt; i <= lp->rows; i++)
	  sprintf(lp->row_name[i], "r_%d", i);

      for(i = lp->sum; i > lp->rows; i--)
	{
	  lp->orig_upbo[i] = lp->orig_upbo[i - rcnt];
	  lp->orig_lowbo[i] = lp->orig_lowbo[i - rcnt];
	  lp->basis[i] = lp->basis[i - rcnt];
	  lp->lower[i] = lp->lower[i - rcnt];
	  lp->must_be_int[i] = lp->must_be_int[i - rcnt];
	}

      for(i = 0; i < rcnt; i++)
	switch(ctype[i])
	  {
	  case REL_LE:
	    lp->ch_sign[lp->rows + 1 - rcnt + i] = FALSE;
	    lp->orig_rh[lp->rows + 1 - rcnt + i] = rh[i];
	    lp->orig_upbo[lp->rows + 1 - rcnt + i] = lp->infinite;
	    lp->orig_lowbo[lp->rows + 1 - rcnt + i] = 0;
	    break;

	  case REL_GE:
	    lp->ch_sign[lp->rows + 1 - rcnt + i] = TRUE;
	    lp->orig_rh[lp->rows + 1 - rcnt + i] = - rh[i];
	    lp->orig_upbo[lp->rows + 1 - rcnt + i] = lp->infinite;
	    lp->orig_lowbo[lp->rows + 1 - rcnt + i] = 0;
	    for(j = rmatbeg[i]; j < rmatbeg[i+1]; j++)
	      rmatval[j] = -rmatval[j];
	    break;

	  case REL_EQ:
	    lp->ch_sign[lp->rows + 1 - rcnt + i] = FALSE;
	    lp->orig_rh[lp->rows + 1 - rcnt + i] = rh[i];
	    lp->orig_upbo[lp->rows + 1 - rcnt + i] = 0;
	    lp->orig_lowbo[lp->rows + 1 - rcnt + i] = 0;
	    break;

	  default:
	    fprintf(stderr, "Wrong constraint type\n");
	    exit (EXIT_FAILURE);
	  }

      for(i = 1; i <= lp->rows - rcnt; i++)
	if(lp->bas[i] > lp->rows - rcnt)
	  lp->bas[i] += rcnt;

      for(i = lp->rows + 1 - rcnt; i <= lp->rows; i++)
	{
	  lp->bas[i] = i;
	  lp->basis[i] = TRUE;
	  lp->lower[i] = TRUE;
	  lp->must_be_int[i] = FALSE;
	}

      lp->row_end_valid = FALSE;

      /* Now integrate the new coefficients into the matrix */
      MALLOC(count, lp->columns);
      MALLOC(ptr1, lp->columns+1);
      MALLOC(ptr2, lp->columns+1);
      MALLOC(tmat, nzcnt);

      /* reorganize as columns... */
      for(i = 0; i < lp->columns; i++)
	count[i] = 0;
      for(i = 0; i < nzcnt; i++)
	++(count[rmatind[i]]);

      mp = tmat;
      for(i = 0; i < lp->columns; i++)
	{
	  ptr1[i] = mp;
	  ptr2[i] = mp;
	  mp += count[i];
	}
      ptr1[lp->columns] = mp;
      ptr2[lp->columns] = mp;

      for(i = 0; i < rcnt; i++)
	for(j = rmatbeg[i]; j < rmatbeg[i+1]; j++)
	  {
	    mp = (ptr2[rmatind[j]])++;
	    mp->row_nr = lp->rows + 1 - rcnt + i;
	    mp->value = rmatval[j];
	  }

      /* merge new columns into matrix */
      dst = lp->mat + lp->non_zeros;
      for(i = lp->columns; i > 0; i--)
	{
	  /* Insert new rows of this column... */
	  save = lp->mat + lp->col_end[i];
	  lp->col_end[i] = dst - lp->mat;
	  src = ptr1[i];
	  end = ptr1[i-1];
	  while (src > end)
	    *--dst = *--src;

	  /* Copy the old rows of this column... */
	  src = save;
	  end = lp->mat + lp->col_end[i-1];
	  while (src > end)
	    *--dst = *--src;
	}

      free(tmat);
      free(ptr2);
      free(ptr1);
      free(count);
    }

  lp->eta_valid = FALSE;
}

void del_constraint(lprec *lp, int del_row)
{
  int i, j;
  unsigned elmnr;
  int startcol;

  if(del_row<1 || del_row>lp->rows)
    {
      fprintf(stderr, "There is no constraint nr. %d\n", del_row);
      exit(EXIT_FAILURE);
    }

  elmnr = 0;
  startcol = 0;

  for(i = 1; i <= lp->columns; i++) {
    for(j = startcol; j < lp->col_end[i]; j++) {
      if(lp->mat[j].row_nr != del_row) {
	lp->mat[elmnr] = lp->mat[j];
	if(lp->mat[elmnr].row_nr > del_row)
	  lp->mat[elmnr].row_nr--;
	elmnr++;
      }
      else
	lp->non_zeros--;
    }
    startcol = lp->col_end[i];
    lp->col_end[i] = elmnr;
  }
  for(i = del_row; i < lp->rows; i++) {
    lp->orig_rh[i] = lp->orig_rh[i + 1];
    lp->ch_sign[i] = lp->ch_sign[i + 1];
    lp->bas[i] = lp->bas[i + 1];
    if(lp->names_used)
      memcpy(lp->row_name[i], lp->row_name[i + 1], NAMELEN);
  }
  for(i = 1; i < lp->rows; i++)
    if(lp->bas[i] >  del_row)
      lp->bas[i]--;

  for(i = del_row; i < lp->sum; i++) {
    lp->lower[i] = lp->lower[i + 1];
    lp->basis[i] = lp->basis[i + 1];
    lp->orig_upbo[i] = lp->orig_upbo[i + 1];
    lp->orig_lowbo[i] = lp->orig_lowbo[i + 1];
    lp->must_be_int[i] = lp->must_be_int[i + 1];
    if(lp->scaling_used)
      lp->scale[i] = lp->scale[i + 1];
  }

  lp->rows--;
  lp->sum--;

  lp->row_end_valid = FALSE;
  lp->eta_valid     = FALSE;
  lp->basis_valid   = FALSE; 
}

void delete_row_set(lprec *lp, int *row_flags)
{
  int i, j, k;
  unsigned elmnr;
  int startcol;
  int * renum;
  int firstrow;
  int num_del;

  /* Find lowest numbered row to delete. */
  firstrow = -1;
  for (i = 0; i <= lp->rows; i++)
    if(row_flags[i])
      {
	firstrow = i;
	break;
      }

  if(firstrow < 0)
    return;

  /* Build a map to renumber the rows and columns. */
  CALLOC(renum, lp->sum + 1);
  num_del = 0;
  j = 0;
  for(i = 0; i <= lp->rows; i++)
    if(row_flags[i])
      {
	renum[i] = -1;
	++num_del;
      }
    else
      renum[i] = j++;
  for(; i <= lp->sum; i++)
    renum[i] = j++;

  elmnr=0;
  startcol=0;

  for(i = 1; i <= lp->columns; i++)
    {
      for(j=startcol; j < lp->col_end[i]; j++)
	{
	  k = renum[lp->mat[j].row_nr];
	  if(k >= 0)
	    {
	      lp->mat[elmnr].row_nr = k;
	      lp->mat[elmnr].value  = lp->mat[j].value;
	      elmnr++;
	    }
	  else
	    lp->non_zeros--;
	}
      startcol=lp->col_end[i];
      lp->col_end[i]=elmnr;
    }
  for(i = firstrow + 1; i <= lp->rows; i++)
    {
      k = renum[i];
      if (k < 0) continue;
      lp->orig_rh[k] = lp->orig_rh[i];
      lp->ch_sign[k] = lp->ch_sign[i];
      if(lp->names_used)
        memcpy(lp->row_name[k], lp->row_name[i], NAMELEN);
    }
  for(i = 1; i <= lp->rows; i++)
    {
      j = renum[i];
      if(j < 0)
	{
	  lp->basis[lp->bas[i]] = 0;
	  lp->lower[lp->bas[i]] = 1;
	  continue;
	}
      k = renum[lp->bas[i]];
      if (k < 0)
	{
	  lp->basis_valid = 0;
	  break;
	}
      lp->bas[j] = k;
    }

  for(i = firstrow + 1; i <= lp->sum; i++)
    {
      k = renum[i];
      if(k < 0) continue;
      lp->lower[k]=lp->lower[i];
      lp->basis[k]=lp->basis[i];
      lp->orig_upbo[k]=lp->orig_upbo[i];
      lp->orig_lowbo[k]=lp->orig_lowbo[i];
      lp->must_be_int[k]=lp->must_be_int[i];
      if(lp->scaling_used)
        lp->scale[k]=lp->scale[i];
    }

  free(renum);

  lp->rows -= num_del;
  lp->sum -= num_del;;

  lp->row_end_valid=FALSE;
  lp->eta_valid=FALSE;
}

void add_lag_con(lprec *lp, REAL *row, short con_type, REAL rhs)
{
  int i;
  REAL sign;
  if(con_type == REL_LE || con_type == REL_EQ)
    sign = 1;
  else if(con_type == REL_GE)
    sign = -1;
  else
    error("con_type not implemented\n");

  lp->nr_lagrange++;
  if(lp->nr_lagrange == 1) {
    CALLOC(lp->lag_row, lp->nr_lagrange);
    CALLOC(lp->lag_rhs, lp->nr_lagrange);
    CALLOC(lp->lambda, lp->nr_lagrange);
    CALLOC(lp->lag_con_type, lp->nr_lagrange);
  }
  else {
    REALLOC(lp->lag_row, lp->nr_lagrange);
    REALLOC(lp->lag_rhs, lp->nr_lagrange);
    REALLOC(lp->lambda, lp->nr_lagrange);
    REALLOC(lp->lag_con_type, lp->nr_lagrange);
  }
  CALLOC(lp->lag_row[lp->nr_lagrange-1], lp->columns+1);
  lp->lag_rhs[lp->nr_lagrange-1] = rhs * sign;
  for( i = 1; i <= lp->columns; i++)
    lp->lag_row[lp->nr_lagrange-1][i] = row[i] * sign;
  lp->lambda[lp->nr_lagrange-1] = 0;
  lp->lag_con_type[lp->nr_lagrange-1]=(con_type == REL_EQ);
}

void str_add_lag_con(lprec *lp, char *row, short con_type, REAL rhs)
{
  int  i;
  REAL *a_row;
  char *p, *new_p;
  CALLOC(a_row, lp->columns + 1);
  p = row;
 
  for(i = 1; i <= lp->columns; i++) {
    a_row[i] = (REAL) strtod(p, &new_p);
    if(p == new_p)
      error("Bad string in str_add_lag_con");
    else
      p = new_p; 
  }
  add_lag_con(lp, a_row, con_type, rhs);
  free(a_row);
}


void add_column(lprec *lp, REAL *column)
{
  int i, elmnr;

  /* if the column has only one entry, this should be handled as a bound, but
     this currently is not the case */

  lp->columns++;
  lp->sum++;
  inc_col_space(lp);
  inc_mat_space(lp, lp->rows + 1);

  if(lp->scaling_used) {
    for(i = 0; i <= lp->rows; i++)
      column[i] *= lp->scale[i];
    lp->scale[lp->sum] = 1;
  }

  elmnr = lp->col_end[lp->columns - 1];
  for(i = 0 ; i <= lp->rows ; i++)
    if(column[i] != 0) {
      lp->mat[elmnr].row_nr = i;
      if(lp->ch_sign[i])
	lp->mat[elmnr].value = -column[i];
      else
	lp->mat[elmnr].value = column[i];
      lp->non_zeros++;
      elmnr++;
    }
  lp->col_end[lp->columns] = elmnr;
  lp->orig_lowbo[lp->sum] = 0;
  lp->orig_upbo[lp->sum] = lp->infinite;
  lp->lower[lp->sum] = TRUE;
  lp->basis[lp->sum] = FALSE;
  lp->must_be_int[lp->sum] = FALSE;
  if(lp->names_used)
    sprintf(lp->col_name[lp->columns], "var_%d", lp->columns);
 
  lp->row_end_valid = FALSE;
}

void str_add_column(lprec *lp, char *col_string)
{
  int  i;
  REAL *aCol;
  char *p, *newp;
  CALLOC(aCol, lp->rows + 1);
  p = col_string;
 
  for(i = 0; i <= lp->rows; i++) {
    aCol[i] = (REAL) strtod(p, &newp);
    if(p == newp)
      error("Bad string in str_add_column");
    else
      p = newp; 
  }
  add_column(lp, aCol);
  free(aCol);
}

void del_column(lprec *lp, int column)
{
  int i, j, from_elm, to_elm, elm_in_col;
  if(column > lp->columns || column < 1)
    error("Column out of range in del_column");
  for(i = 1; i <= lp->rows; i++) {
    if(lp->bas[i] == lp->rows + column)
      lp->basis_valid = FALSE;
    else if(lp->bas[i] > lp->rows + column)
      lp->bas[i]--;
  }
  for(i = lp->rows + column; i < lp->sum; i++) {
    if(lp->names_used)
      memcpy(lp->col_name[i - lp->rows], lp->col_name[i - lp->rows + 1], NAMELEN);
    lp->must_be_int[i] = lp->must_be_int[i + 1];
    lp->orig_upbo[i] = lp->orig_upbo[i + 1];
    lp->orig_lowbo[i] = lp->orig_lowbo[i + 1];
    lp->upbo[i] = lp->upbo[i + 1];
    lp->lowbo[i] = lp->lowbo[i + 1];
    lp->basis[i] = lp->basis[i + 1];
    lp->lower[i] = lp->lower[i + 1];
    if(lp->scaling_used)
      lp->scale[i] = lp->scale[i + 1];
  }
  for(i = 0; i < lp->nr_lagrange; i++)
    for(j = column; j <= lp->columns; j++)
      lp->lag_row[i][j] = lp->lag_row[i][j+1];
  to_elm = lp->col_end[column-1];
  from_elm = lp->col_end[column];
  elm_in_col = from_elm-to_elm;
  for(i = from_elm; i < lp->non_zeros; i++) {
    lp->mat[to_elm] = lp->mat[i];
    to_elm++;
  }
  for(i = column; i < lp->columns; i++)
    lp->col_end[i] = lp->col_end[i + 1] - elm_in_col;
  lp->non_zeros -= elm_in_col;
  lp->row_end_valid = FALSE;
  lp->eta_valid = FALSE;

  lp->sum--;
  lp->columns--;
}

void set_upbo(lprec *lp, int column, REAL value)
{
  if(column > lp->columns || column < 1)
    error("Column out of range");
  if(lp->scaling_used)
    value /= lp->scale[lp->rows + column];
  if(value < lp->orig_lowbo[lp->rows + column])
    error("Upperbound must be >= lowerbound"); 
  lp->eta_valid = FALSE;
  lp->orig_upbo[lp->rows + column] = value;
}

void set_lowbo(lprec *lp, int column, REAL value)
{
  if(column > lp->columns || column < 1)
    error("Column out of range");
  if(lp->scaling_used)
    value /= lp->scale[lp->rows + column];
  if(value > lp->orig_upbo[lp->rows + column])
    error("Upperbound must be >= lowerbound"); 
  /*
    if(value < 0)
    error("Lower bound cannot be < 0");
    */
  lp->eta_valid = FALSE;
  lp->orig_lowbo[lp->rows + column] = value;
}

void set_bounds(lprec *lp, int column, REAL lower, REAL upper)
{
  if(column > lp->columns || column < 1)
    error("Column out of range");
  if(lp->scaling_used)
    {
      lower /= lp->scale[lp->rows + column];
      upper /= lp->scale[lp->rows + column];
    }
  if(lower > upper)
    error("Upperbound must be >= lowerbound");
  lp->eta_valid = FALSE;
  lp->orig_lowbo[lp->rows+column] = lower;
  lp->orig_upbo[lp->rows+column] = upper;
}

void set_int(lprec *lp, int column, short must_be_int)
{
  if(column > lp->columns || column < 1)
    error("Column out of range");
  lp->must_be_int[lp->rows + column] = must_be_int;
  if(must_be_int == TRUE)
    if(lp->columns_scaled)
      unscale_columns(lp);
}

void set_rh(lprec *lp, int row, REAL value)
{
  if(row > lp->rows || row < 0)
    error("Row out of Range");

  if ((row == 0) && (!lp->maximise))  /* setting of RHS of OF IS meaningful */
    value = -value;
  if(lp->scaling_used) {
    if(lp->ch_sign[row])
      lp->orig_rh[row] = -value * lp->scale[row];
    else
      lp->orig_rh[row] = value * lp->scale[row];
  }
  else
    if(lp->ch_sign[row])
      lp->orig_rh[row] = -value;
    else
      lp->orig_rh[row] = value;
  lp->eta_valid = FALSE;
} 

void set_rh_vec(lprec *lp, REAL *rh)
{
  int i;
  if(lp->scaling_used) {
    for(i = 1; i <= lp->rows; i++)
      if(lp->ch_sign[i])
        lp->orig_rh[i] = -rh[i]*lp->scale[i];
      else
        lp->orig_rh[i] = rh[i]*lp->scale[i];
  }
  else
    for(i = 1; i <= lp->rows; i++)
      if(lp->ch_sign[i])
        lp->orig_rh[i] = -rh[i];
      else
        lp->orig_rh[i] = rh[i];
  lp->eta_valid = FALSE;
}

void str_set_rh_vec(lprec *lp, char *rh_string)
{
  int  i;
  REAL *newrh;
  char *p, *newp;
  CALLOC(newrh, lp->rows + 1);
  p = rh_string;
 
  for(i = 1; i <= lp->rows; i++) {
    newrh[i] = (REAL) strtod(p, &newp);
    if(p == newp)
      error("Bad string in str_set_rh_vec");
    else
      p = newp; 
  }
  set_rh_vec(lp, newrh);
  free(newrh);
}


void set_maxim(lprec *lp)
{
  int i;
  if(lp->maximise == FALSE) {
    for(i = 0; i < lp->non_zeros; i++)
      if(lp->mat[i].row_nr == 0)
	lp->mat[i].value *= -1;
    lp->eta_valid = FALSE;
    lp->orig_rh[0] *= -1;
  } 
  lp->maximise = TRUE;
  lp->ch_sign[0] = TRUE;
}

void set_minim(lprec *lp)
{
  int i;
  if(lp->maximise == TRUE) {
    for(i = 0; i < lp->non_zeros; i++)
      if(lp->mat[i].row_nr == 0)
	lp->mat[i].value = -lp->mat[i].value;
    lp->eta_valid = FALSE;
    lp->orig_rh[0] *= -1;
  } 
  lp->maximise = FALSE;
  lp->ch_sign[0] = FALSE;
}

void set_constr_type(lprec *lp, int row, short con_type)
{
  int i;
  if(row > lp->rows || row < 1)
    error("Row out of Range");
  if(con_type == REL_EQ) {
    lp->orig_upbo[row] = 0;
    lp->basis_valid = FALSE;
    if(lp->ch_sign[row]) {
      for(i = 0; i < lp->non_zeros; i++)
	if(lp->mat[i].row_nr == row)
	  lp->mat[i].value *= -1;
      lp->eta_valid = FALSE;
      lp->ch_sign[row] = FALSE;
      if(lp->orig_rh[row] != 0)
	lp->orig_rh[row] *= -1;
    }
  }
  else if(con_type == REL_LE) {
    lp->orig_upbo[row] = lp->infinite;
    lp->basis_valid = FALSE;
    if(lp->ch_sign[row]) {
      for(i = 0; i < lp->non_zeros; i++)
	if(lp->mat[i].row_nr == row)
	  lp->mat[i].value *= -1;
      lp->eta_valid = FALSE;
      lp->ch_sign[row] = FALSE;
      if(lp->orig_rh[row] != 0)
	lp->orig_rh[row] *= -1;
    }
  }
  else if(con_type == REL_GE) {
    lp->orig_upbo[row] = lp->infinite;
    lp->basis_valid = FALSE;
    if(!lp->ch_sign[row]) {
      for(i = 0; i < lp->non_zeros; i++)
	if(lp->mat[i].row_nr == row)
	  lp->mat[i].value *= -1;
      lp->eta_valid = FALSE;
      lp->ch_sign[row] = TRUE;
      if(lp->orig_rh[row] != 0)
	lp->orig_rh[row] *= -1;
    }
  } 
  else
    error("Constraint type not (yet) implemented");
}

REAL mat_elm(lprec *lp, int row, int column)
{
  REAL value;
  int elmnr;
  if(row < 0 || row > lp->rows)
    error("Row out of range in mat_elm");
  if(column < 1 || column > lp->columns)
    error("Column out of range in mat_elm");
  value = 0;
  elmnr = lp->col_end[column-1];
  while(lp->mat[elmnr].row_nr != row && elmnr < lp->col_end[column])
    elmnr++;
  if(elmnr != lp->col_end[column]) {
    value = lp->mat[elmnr].value;
    if(lp->ch_sign[row])
      value = -value;
    if(lp->scaling_used)
      value /= lp->scale[row] * lp->scale[lp->rows + column];
  }
  return(value);
}


void get_row(lprec *lp, int row_nr, REAL *row)
{
  int i, j;

  if(row_nr <0 || row_nr > lp->rows)
    error("Row nr. out of range in get_row");
  for(i = 1; i <= lp->columns; i++) {
    row[i] = 0;
    for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
      if(lp->mat[j].row_nr == row_nr)
	row[i] = lp->mat[j].value;
    if(lp->scaling_used)
      row[i] /= lp->scale[lp->rows + i] * lp->scale[row_nr];
  }
  if(lp->ch_sign[row_nr])
    for(i = 0; i <= lp->columns; i++)
      if(row[i] != 0)
        row[i] = -row[i];
}

void get_column(lprec *lp, int col_nr, REAL *column)
{
  int i;

  if(col_nr < 1 || col_nr > lp->columns)
    error("Col. nr. out of range in get_column");
  for(i = 0; i <= lp->rows; i++)
    column[i] = 0;
  for(i = lp->col_end[col_nr - 1]; i < lp->col_end[col_nr]; i++)
    column[lp->mat[i].row_nr] = lp->mat[i].value;
  for(i = 0; i <= lp->rows; i++)
    if(column[i] != 0) {
      if(lp->ch_sign[i])
	column[i] *= -1;
      if(lp->scaling_used)
	column[i] /= (lp->scale[i] * lp->scale[lp->rows + col_nr]);
    }
}

void get_reduced_costs(lprec *lp, REAL *rc)
{
  int varnr, i, j, row;
  REAL f;

  if(!lp->basis_valid)
    error("Not a valid basis in get_reduced_costs");

  if(!lp->eta_valid)
    invert(lp);  

  for(i = 1; i <= lp->sum; i++)
    rc[i] = 0;
  rc[0] = 1;

  btran(lp, rc);

  for(i = 1; i <= lp->columns; i++) {
    varnr = lp->rows + i;
    if(!lp->basis[varnr])
      if(lp->upbo[varnr] > 0) {
	if(lp->scaling_used) {
	  f = 0;
	  for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
	    row = lp->mat[j].row_nr;
	    f += rc[row] * lp->mat[j].value
		  * (lp->scale[row] * lp->scale[varnr]);
	  }
	  rc[varnr] = f;
	}
	else {
	  f = 0;
	  for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
	    f += rc[lp->mat[j].row_nr] * lp->mat[j].value;
	  rc[varnr] = f;
	}
      }
  }
  for(i = 1; i <= lp->sum; i++)
    my_round(rc[i], lp->epsd);
}

void get_slack_vars(lprec *lp, REAL *slacks)
{
  int    i, basi;

  if(!lp->basis_valid)
    error("Not a valid basis in get_slack_vars");

  /* zero all results of rows */
  memset(slacks, '\0', (lp->rows + 1) * sizeof(REAL));

  /* Lower bounds are always zero for slack variables.  Their upper	*/
  /* bounds are always infinity (except for equality constraints).	*/
  /* However they NEVER appear out of the basis except at their lower	*/
  /* bounds.								*/

  if(lp->scaling_used) {
    for(i = 1; i <= lp->rows; i++) {
      basi = lp->bas[i];
      if(basi <= lp->rows)
	slacks[basi] = lp->rhs[i] * lp->scale[basi];
    }
  }
  else {
    for(i = 1; i <= lp->rows; i++) {
      basi = lp->bas[i];
      if(basi <= lp->rows)
	slacks[basi] = lp->rhs[i];
    }
  }
}

short is_feasible(lprec *lp, REAL *values)
{
  int i, elmnr;
  REAL *this_rhs;
  REAL dist;

  if(lp->scaling_used) {
    for(i = lp->rows + 1; i <= lp->sum; i++)
      if(   values[i - lp->rows] < lp->orig_lowbo[i] * lp->scale[i]
	 || values[i - lp->rows] > lp->orig_upbo[i]  * lp->scale[i])
	return(FALSE);
  }
  else {
    for(i = lp->rows + 1; i <= lp->sum; i++)
      if(   values[i - lp->rows] < lp->orig_lowbo[i]
	 || values[i - lp->rows] > lp->orig_upbo[i])
	return(FALSE);
  }
  CALLOC(this_rhs, lp->rows + 1);
    for(i = 1; i <= lp->columns; i++)
      for(elmnr = lp->col_end[i - 1]; elmnr < lp->col_end[i]; elmnr++)
	this_rhs[lp->mat[elmnr].row_nr] += lp->mat[elmnr].value * values[i]; 
  for(i = 1; i <= lp->rows; i++) {
    dist = lp->orig_rh[i] - this_rhs[i];
    my_round(dist, 0.001) /* ugly constant, MB */
      if((lp->orig_upbo[i] == 0 && dist != 0) || dist < 0) {
	free(this_rhs);
	return(FALSE);
      }     
  } 
  free(this_rhs);
  return(TRUE);
}

/* fixed by Enrico Faggiolo */
short column_in_lp(lprec *lp, REAL *testcolumn)
{
  int i, j;
  int nz, ident;
  REAL value;

  for(nz = 0, i = 0; i <= lp->rows; i++)
    if(my_abs(testcolumn[i]) > lp->epsel) nz++;

  if(lp->scaling_used)
    for(i = 1; i <= lp->columns; i++) {
      ident = nz;
      for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
	value = lp->mat[j].value;
	if(lp->ch_sign[lp->mat[j].row_nr])
	  value = -value;
	value /= lp->scale[lp->rows + i];
	value /= lp->scale[lp->mat[j].row_nr];
	value -= testcolumn[lp->mat[j].row_nr];
	if(my_abs(value) > lp->epsel)
	  break;
	ident--;
	if(ident == 0)
	  return(TRUE);
      }
    }
  else
    for(i = 1; i <= lp->columns; i++) {
      ident = nz;
      for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
	value = lp->mat[j].value;
	if(lp->ch_sign[lp->mat[j].row_nr])
	  value = -value;
	value -= testcolumn[lp->mat[j].row_nr];
	if(my_abs(value) > lp->epsel)
	  break;
	ident--;
	if(ident == 0)
	  return(TRUE);
      }
    }
  return(FALSE);
}

void print_lp(lprec *lp)
{
  int i, j;
  REAL *fatmat;
  CALLOC(fatmat, (lp->rows + 1) * lp->columns);
  for(i = 1; i <= lp->columns; i++)
    for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
      fatmat[(i - 1) * (lp->rows + 1) + lp->mat[j].row_nr] = lp->mat[j].value;

  printf("problem name: %s\n", lp->lp_name);
  printf("          ");
  for(j = 1; j <= lp->columns; j++)
    if(lp->names_used)
      printf("%8s ", lp->col_name[j]);
    else
      printf("Var[%3d] ", j);
  if(lp->maximise)
    {
      printf("\nMaximise  ");
      for(j = 0; j < lp->columns; j++)
	printf("%8g ", (double) -fatmat[j * (lp->rows + 1)]);
    }
  else
    {
      printf("\nMinimize  ");
      for(j = 0; j < lp->columns; j++)
	printf("%8g ", (double) fatmat[j * (lp->rows + 1)]);
    }
  printf("\n");
  for(i = 1; i <= lp->rows; i++) {
    if(lp->names_used)
      printf("%-9s ", lp->row_name[i]);
    else
      printf("Row[%3d]  ", i);
    for(j = 0; j < lp->columns; j++)
      if(lp->ch_sign[i] && fatmat[j*(lp->rows + 1) + i] != 0)
	printf("%8g ", (double)-fatmat[j * (lp->rows+1) + i]);
      else
	printf("%8g ", (double)fatmat[j * (lp->rows + 1) + i]);
    if(lp->orig_upbo[i] != 0) {
      if(lp->ch_sign[i])
	printf(">= ");
      else
	printf("<= ");
    }
    else
      printf(" = ");
    if(lp->ch_sign[i])
      printf("%8g", (double)-lp->orig_rh[i]);
    else
      printf("%8g", (double)lp->orig_rh[i]);
    if(lp->orig_lowbo[i] != 0) {
      printf("  %s = %8g", (lp->ch_sign[i]) ? "lowbo" : "upbo",
	     (double)lp->orig_lowbo[i]);
    }
    if((lp->orig_upbo[i] != lp->infinite) && (lp->orig_upbo[i] != 0.0)) {
      printf("  %s = %8g", (lp->ch_sign[i]) ? "upbo" : "lowbo",
	     (double)lp->orig_upbo[i]);
    }
    printf("\n");
  }
  printf("Type      ");
  for(i = 1; i <= lp->columns; i++)
    if(lp->must_be_int[lp->rows + i] == TRUE)
      printf("     Int ");
    else
      printf("    Real ");
  printf("\nupbo      ");
  for(i = 1; i <= lp->columns; i++)
    if(lp->orig_upbo[lp->rows + i] == lp->infinite)
      printf(" Infinite");
    else
      printf("%8g ", (double)lp->orig_upbo[lp->rows + i]);
  printf("\nlowbo     ");
  for(i = 1; i <= lp->columns; i++)
    printf("%8g ", (double)lp->orig_lowbo[lp->rows + i]);
  printf("\n");
  for(i = 0; i < lp->nr_lagrange; i++) {
    printf("lag[%d]  ", i);
    for(j = 1; j <= lp->columns; j++)
      printf("%8g ", (double)lp->lag_row[i][j]);
    if(lp->orig_upbo[i] == lp->infinite) {
      if(lp->lag_con_type[i] == REL_GE)
	printf(">= ");
      else if(lp->lag_con_type[i] == REL_LE)
	printf("<= ");
      else if(lp->lag_con_type[i] == REL_EQ)
	printf(" = ");
    }
    printf("%8g\n", (double)lp->lag_rhs[i]);
  }

  free(fatmat);
}  

void set_row_name(lprec *lp, int row, nstring new_name)
{
  int i;
  hashelem *hp;

  if(!lp->names_used) {
    CALLOC(lp->row_name, lp->rows_alloc + 1);
    CALLOC(lp->col_name, lp->columns_alloc + 1);
    lp->names_used = TRUE;
    for(i = 0; i <= lp->rows; i++)
      sprintf(lp->row_name[i], "r_%d", i);
    for(i = 1; i <= lp->columns; i++)
      sprintf(lp->col_name[i], "var_%d", i);
  }
  strcpy(lp->row_name[row], new_name);
  hp = puthash(lp->row_name[row], lp->rowname_hashtab);
  hp->index = row;
}

void set_col_name(lprec *lp, int column, nstring new_name)
{
  int i;
  hashelem *hp;
 
  if(!lp->names_used) {
    CALLOC(lp->row_name, lp->rows_alloc + 1);
    CALLOC(lp->col_name, lp->columns_alloc + 1);
    lp->names_used = TRUE;
    for(i = 0; i <= lp->rows; i++)
      sprintf(lp->row_name[i], "r_%d", i);
    for(i = 1; i <= lp->columns; i++)
      sprintf(lp->col_name[i], "var_%d", i);
  }
  strcpy(lp->col_name[column], new_name);
  hp = puthash(lp->col_name[column], lp->colname_hashtab);
  hp->index = column;
}

static REAL minmax_to_scale(REAL min, REAL max)
{
  REAL scale;

  /* should do something sensible when min or max is 0, MB */
  if((min == 0) || (max == 0))
    return((REAL)1);

  scale = 1 / pow(10.0, (log10(min) + log10(max)) / 2);
  return(scale);
}

void unscale_columns(lprec *lp)
{
  int i, j;

  /* unscale mat */
  for(j = 1; j <= lp->columns; j++)
    for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
      lp->mat[i].value /= lp->scale[lp->rows + j];

  /* unscale bounds as well */
  for(i = lp->rows + 1; i <= lp->sum; i++) { /* was < */ /* changed by PN */
    if(lp->orig_lowbo[i] != 0)
      lp->orig_lowbo[i] *= lp->scale[i];
    if(lp->orig_upbo[i] != lp->infinite)
      lp->orig_upbo[i] *= lp->scale[i];
  }
    
  for(i = lp->rows + 1; i<= lp->sum; i++)
    lp->scale[i] = 1;
  lp->columns_scaled = FALSE;
  lp->eta_valid = FALSE;
}

void unscale(lprec *lp)
{
  int i, j;
  
  if(lp->scaling_used) {
    /* unscale mat */
    for(j = 1; j <= lp->columns; j++)
      for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
	lp->mat[i].value /= lp->scale[lp->rows + j];

    /* unscale bounds */
    for(i = lp->rows + 1; i <= lp->sum; i++) { /* was < */ /* changed by PN */
      if(lp->orig_lowbo[i] != 0)
	lp->orig_lowbo[i] *= lp->scale[i];
      if(lp->orig_upbo[i] != lp->infinite)
	lp->orig_upbo[i] *= lp->scale[i];
    }
    
    /* unscale the matrix */
    for(j = 1; j <= lp->columns; j++)
      for(i = lp->col_end[j-1]; i < lp->col_end[j]; i++)
	lp->mat[i].value /= lp->scale[lp->mat[i].row_nr];

    /* unscale the rhs! */
    for(i = 0; i <= lp->rows; i++)
      lp->orig_rh[i] /= lp->scale[i];

    /* and don't forget to unscale the upper and lower bounds ... */
    for(i = 0; i <= lp->rows; i++) {
      if(lp->orig_lowbo[i] != 0)
	lp->orig_lowbo[i] /= lp->scale[i];
      if(lp->orig_upbo[i] != lp->infinite)
	lp->orig_upbo[i] /= lp->scale[i];
    }

    free(lp->scale);
    lp->scaling_used = FALSE;
    lp->eta_valid = FALSE;
  }
}


void auto_scale(lprec *lp)
{
  int i, j, row_nr;
  REAL *row_max, *row_min, *scalechange, absval;
  REAL col_max, col_min;
  
  if(!lp->scaling_used) {
    MALLOC(lp->scale, lp->sum_alloc + 1);
    for(i = 0; i <= lp->sum; i++)
      lp->scale[i] = 1;
  }
  
  MALLOC(row_max, lp->rows + 1);
  MALLOC(row_min, lp->rows + 1);
  MALLOC(scalechange, lp->sum + 1);

  /* initialise min and max values */
  for(i = 0; i <= lp->rows; i++) {
    row_max[i] = 0;
    row_min[i] = lp->infinite;
  }

  /* calculate min and max absolute values of rows */
  for(j = 1; j <= lp->columns; j++)
    for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++) {
      row_nr = lp->mat[i].row_nr;
      absval = my_abs(lp->mat[i].value);
      if(absval != 0) {
	row_max[row_nr] = my_max(row_max[row_nr], absval);
	row_min[row_nr] = my_min(row_min[row_nr], absval);
      }
    }    
  /* calculate scale factors for rows */
  for(i = 0; i <= lp->rows; i++) {
    scalechange[i] = minmax_to_scale(row_min[i], row_max[i]);
    lp->scale[i] *= scalechange[i];
  }

  /* now actually scale the matrix */
  for(j = 1; j <= lp->columns; j++)
    for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
      lp->mat[i].value *= scalechange[lp->mat[i].row_nr];

  /* and scale the rhs and the row bounds (RANGES in MPS!!) */
  for(i = 0; i <= lp->rows; i++) {
    lp->orig_rh[i] *= scalechange[i];

    if((lp->orig_upbo[i] < lp->infinite) && (lp->orig_upbo[i] != 0))
      lp->orig_upbo[i] *= scalechange[i];

    if(lp->orig_lowbo[i] != 0)
      lp->orig_lowbo[i] *= scalechange[i];
  }

  free(row_max);
  free(row_min);
  
  /* calculate column scales */
  for(j = 1; j <= lp->columns; j++) {
    if(lp->must_be_int[lp->rows + j]) { /* do not scale integer columns */
      scalechange[lp->rows + j] = 1;
    }
    else {
      col_max = 0;
      col_min = lp->infinite;
      for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++) {
	if(lp->mat[i].value != 0) {
	  col_max = my_max(col_max, my_abs(lp->mat[i].value));
	  col_min = my_min(col_min, my_abs(lp->mat[i].value));
	}
      }
      scalechange[lp->rows + j]  = minmax_to_scale(col_min, col_max);
      lp->scale[lp->rows + j] *= scalechange[lp->rows + j];
    }
  }
  
  /* scale mat */
  for(j = 1; j <= lp->columns; j++)
    for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
      lp->mat[i].value *= scalechange[lp->rows + j];
  
  /* scale bounds as well */
  for(i = lp->rows + 1; i <= lp->sum; i++) { /* was < *//* changed by PN */
    if(lp->orig_lowbo[i] != 0)
      lp->orig_lowbo[i] /= scalechange[i];
    if(lp->orig_upbo[i] != lp->infinite)
      lp->orig_upbo[i] /= scalechange[i];
  }
  lp->columns_scaled = TRUE;

  free(scalechange);
  lp->scaling_used = TRUE;
  lp->eta_valid = FALSE;
}

void reset_basis(lprec *lp)
{
  lp->basis_valid = FALSE;
}

void print_solution(lprec *lp)
{
  int i;
  FILE *stream;

  stream = stdout;

  fprintf(stream, "Value of objective function: %g\n",
	  (double)lp->best_solution[0]);

  /* print normal variables */
  for(i = 1; i <= lp->columns; i++)
    if(lp->names_used)
      fprintf(stream, "%-20s %g\n", lp->col_name[i],
	      (double)lp->best_solution[lp->rows + i]);
    else
      fprintf(stream, "Var [%d] %g\n", i,
	      (double)lp->best_solution[lp->rows + i]);

  /* print achieved constraint values */
  if(lp->verbose) {
    fprintf(stream, "\nActual values of the constraints:\n");
    for(i = 1; i <= lp->rows; i++)
      if(lp->names_used)
	fprintf(stream, "%-20s %g\n", lp->row_name[i],
		(double)lp->best_solution[i]);
      else
	fprintf(stream, "Row [%d] %g\n", i,
		(double)lp->best_solution[i]);  
  }

  if((lp->verbose || lp->print_duals)) {
    if(lp->max_level != 1)
      fprintf(stream,
	      "These are the duals from the node that gave the optimal solution.\n");
    else
      fprintf(stream, "\nDual values:\n");
    for(i = 1; i <= lp->rows; i++)
      if(lp->names_used)
	fprintf(stream, "%-20s %g\n", lp->row_name[i],
		(double)lp->duals[i]);
      else
	fprintf(stream, "Row [%d] %g\n", i, (double)lp->duals[i]); 
  }
  fflush(stream);
} /* Printsolution */

void write_LP(lprec *lp, FILE *output)
{
  int i, j;
  REAL *row;
  
  MALLOC(row, lp->columns+1);
  if(lp->maximise)
    fprintf(output, "max:");
  else
    fprintf(output, "min:");

  get_row(lp, 0, row);
  for(i = 1; i <= lp->columns; i++)
    if(row[i] != 0) {
      if(row[i] == -1)
	fprintf(output, " -");
      else if(row[i] == 1)
	fprintf(output, " +");
      else 
	fprintf(output, " %+g ", (double)row[i]);
      if(lp->names_used)
	fprintf(output, "%s", lp->col_name[i]);
      else
	fprintf(output, "x%d", i);
    }
  fprintf(output, ";\n");

  for(j = 1; j <= lp->rows; j++) {
    if(lp->names_used)
      fprintf(output, "%s:", lp->row_name[j]);
    get_row(lp, j, row);
    for(i = 1; i <= lp->columns; i++)
      if(row[i] != 0) {
	if(row[i] == -1)
	  fprintf(output, " -");
	else if(row[i] == 1)
	  fprintf(output, " +");
	else 
	  fprintf(output, " %+g ", (double)row[i]);
	if(lp->names_used)
	  fprintf(output, "%s", lp->col_name[i]);
	else
	  fprintf(output, "x%d", i);
      }
    if(lp->orig_upbo[j] == 0)
      fprintf(output, " =");
    else if(lp->ch_sign[j])
      fprintf(output, " >");
    else
      fprintf(output, " <");
    if(lp->ch_sign[j])
      fprintf(output, " %16g;\n", (double)-lp->orig_rh[j]);
    else
      fprintf(output, " %16g;\n", (double)lp->orig_rh[j]);
  }
  for(i = lp->rows + 1; i <= lp->sum; i++) {
    if(lp->orig_lowbo[i] != 0) {
      if(lp->names_used)
	fprintf(output, "%s > %16g;\n", lp->col_name[i - lp->rows],
		(double)lp->orig_lowbo[i]);
      else
	fprintf(output, "x%d > %16g;\n", i - lp->rows,
		(double)lp->orig_lowbo[i]);
    }
    if(lp->orig_upbo[i] != lp->infinite) {
      if(lp->names_used)
	fprintf(output, "%s < %16g;\n", lp->col_name[i - lp->rows],
		(double)lp->orig_upbo[i]);
      else
	fprintf(output, "x%d < %16g;\n", i - lp->rows,
		(double)lp->orig_upbo[i]);
    }
  }


  i = 1;
  while(!lp->must_be_int[lp->rows + i]  && i <= lp->columns)
    i++;
  if(i <= lp->columns) {
    if(lp->names_used)  
      fprintf(output, "\nint %s", lp->col_name[i]);
    else
      fprintf(output, "\nint x%d", i);
    i++;
    for(; i <= lp->columns; i++)
      if(lp->must_be_int[lp->rows + i]) {
	if(lp->names_used)  
	  fprintf(output, ",%s", lp->col_name[i]);
	else
	  fprintf(output, ", x%d", i);
      }
    fprintf(output, ";\n");
  }
  free(row);
}




void write_MPS(lprec *lp, FILE *output)
{
  int i, j, marker, putheader;
  REAL *column, a;


  MALLOC(column, lp->rows + 1);
  marker = 0;   
  fprintf(output, "NAME          %s\n", lp->lp_name);
  fprintf(output, "ROWS\n");
  for(i = 0; i <= lp->rows; i++) {
    if(i == 0)
      fprintf(output, " N  ");
    else
      if(lp->orig_upbo[i] != 0) {
	if(lp->ch_sign[i])
	  fprintf(output, " G  ");
	else
	  fprintf(output, " L  ");
      }
      else
	fprintf(output, " E  ");
    if(lp->names_used)
      fprintf(output, "%s\n", lp->row_name[i]);
    else
      fprintf(output, "r_%d\n", i);
  }
      
  fprintf(output, "COLUMNS\n");
  j = 0;
  for(i = 1; i <= lp->columns; i++) {
    if((lp->must_be_int[i + lp->rows]) && (marker % 2) == 0) {
      fprintf(output,
	      "    MARK%04d  'MARKER'                 'INTORG'\n",
	      marker);
      marker++;
    }
    if((!lp->must_be_int[i + lp->rows]) && (marker % 2) == 1) {
      fprintf(output,
	      "    MARK%04d  'MARKER'                 'INTEND'\n",
	      marker);
      marker++;
    }
    /* this gets slow for large LP problems. Implement a sparse version? */
    get_column(lp, i, column);
    j = 0;
    if(lp->maximise) {
      if(column[j] != 0) { 
	if(lp->names_used)
	  fprintf(output, "    %-8s  %-8s  %16g\n", lp->col_name[i],
		  lp->row_name[j], (double)-column[j]);
	else
	  fprintf(output, "    var_%-4d  r_%-6d  %16g\n", i, j,
		  (double)-column[j]);
      }
    } 
    else {
      if(column[j] != 0) { 
	if(lp->names_used)
	  fprintf(output, "    %-8s  %-8s  %16g\n", lp->col_name[i],
		  lp->row_name[j], (double)column[j]);
	else
	  fprintf(output, "    var_%-4d  r_%-6d  %16g\n", i, j,
		  (double)column[j]);
      }
    }
    for(j = 1; j <= lp->rows; j++)
      if(column[j] != 0) { 
	if(lp->names_used)
	  fprintf(output, "    %-8s  %-8s  %16g\n", lp->col_name[i],
		  lp->row_name[j], (double)column[j]);
	else
	  fprintf(output, "    var_%-4d  r_%-6d  %16g\n", i, j,
		  (double)column[j]);
      }
  }
  if((marker % 2) == 1) {
    fprintf(output, "    MARK%04d  'MARKER'                 'INTEND'\n",
	    marker);
    marker++;
  }

  fprintf(output, "RHS\n");
  for(i = 1; i <= lp->rows; i++) {
    a = lp->orig_rh[i];
    if(lp->scaling_used)
      a /= lp->scale[i];

    if(lp->ch_sign[i]) {
      if(lp->names_used)
	fprintf(output, "    RHS       %-8s  %16g\n", lp->row_name[i],
		(double)-a);
      else
	fprintf(output, "    RHS       r_%-6d  %16g\n", i, (double)-a);
    }
    else {
      if(lp->names_used)
	fprintf(output, "    RHS       %-8s  %16g\n", lp->row_name[i],
		(double)a);
      else
	fprintf(output, "    RHS       r_%-6d  %16g\n", i, (double)a);
    }
  }

  putheader = TRUE;
  for(i = 1; i <= lp->rows; i++)
    if((lp->orig_upbo[i] != lp->infinite) && (lp->orig_upbo[i] != 0.0)) {
      if(putheader) {
	fprintf(output, "RANGES\n");
	putheader = FALSE;
      }
      a = lp->orig_upbo[i];
      if(lp->scaling_used)
	a /= lp->scale[i];
      if(lp->names_used)
	fprintf(output, "    RGS       %-8s  %16g\n", lp->row_name[i],
		(double)a);
      else
	fprintf(output, "    RGS       r_%-6d  %16g\n", i,
		(double)a);
    }
    else if((lp->orig_lowbo[i] != 0.0)) {
      if(putheader) {
	fprintf(output, "RANGES\n");
	putheader = FALSE;
      }
      a = lp->orig_lowbo[i];
      if(lp->scaling_used)
	a /= lp->scale[i];
      if(lp->names_used)
	fprintf(output, "    RGS       %-8s  %16g\n", lp->row_name[i],
		(double)-a);
      else
	fprintf(output, "    RGS       r_%-6d  %16g\n", i,
		(double)-a);
    }

  fprintf(output, "BOUNDS\n");
  if(lp->names_used)
    for(i = lp->rows + 1; i <= lp->sum; i++) {
      if((lp->orig_lowbo[i] != 0) && (lp->orig_upbo[i] < lp->infinite) &&
	 (lp->orig_lowbo[i] == lp->orig_upbo[i])) {
	a = lp->orig_upbo[i];
	if(lp->scaling_used)
	  a *= lp->scale[i];
	fprintf(output, " FX BND       %-8s  %16g\n",
		lp->col_name[i - lp->rows], (double)a);
      }
      else {
	if(lp->orig_upbo[i] < lp->infinite) {
	  a = lp->orig_upbo[i];
	  if(lp->scaling_used)
	    a *= lp->scale[i];
	  fprintf(output, " UP BND       %-8s  %16g\n",
		  lp->col_name[i - lp->rows], (double)a);
	}
	if(lp->orig_lowbo[i] != 0) {
	  a = lp->orig_lowbo[i];
	  if(lp->scaling_used)
	    a *= lp->scale[i];
	  fprintf(output, " LO BND       %-8s  %16g\n",
		  lp->col_name[i - lp->rows], (double)lp->orig_lowbo[i]);
	}
      }
    }
  else
    for(i = lp->rows + 1; i <= lp->sum; i++) {
      if((lp->orig_lowbo[i] != 0) && (lp->orig_upbo[i] < lp->infinite) &&
	 (lp->orig_lowbo[i] == lp->orig_upbo[i])) {
	a = lp->orig_upbo[i];
	if(lp->scaling_used)
	  a *= lp->scale[i];
	fprintf(output, " FX BND       %-8s  %16g\n",
		lp->col_name[i - lp->rows], (double)a);
      }
      else {
	if(lp->orig_upbo[i] < lp->infinite) {
	  a = lp->orig_upbo[i];
	  if(lp->scaling_used)
	    a *= lp->scale[i];
	  fprintf(output, " UP BND       var_%-4d  %16g\n",
		  i - lp->rows, (double)a);
	}
	if(lp->orig_lowbo[i] != 0) {
	  a = lp->orig_lowbo[i];
	  if(lp->scaling_used)
	    a *= lp->scale[i];
	  fprintf(output, " LO BND       var_%-4d  %16g\n", i - lp->rows,
		  (double)a);
	}
      }
    }
  fprintf(output, "ENDATA\n");
  free(column);
}

void print_duals(lprec *lp)
{
  int i;
  for(i = 1; i <= lp->rows; i++)
    if(lp->names_used)
      fprintf(stdout, "%s [%d] %g\n", lp->row_name[i], i,
	      (double)lp->duals[i]);
    else
      fprintf(stdout, "Dual [%d] %g\n", i, (double)lp->duals[i]);
}

void print_scales(lprec *lp)
{
  int i;
  if(lp->scaling_used) {
    for(i = 0; i <= lp->rows; i++)
      fprintf(stdout, "Row[%d]    scaled at %g\n", i,
	      (double)lp->scale[i]);
    for(i = 1; i <= lp->columns; i++)
      fprintf(stdout, "Column[%d] scaled at %g\n", i,
	      (double)lp->scale[lp->rows + i]);
  }
}

