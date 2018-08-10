#include "common_R.h"
#include "common_debug.h"

extern "C"
SEXP cuR_object_duplicate( SEXP obj_r ){
  SEXP duplicate_r = Rf_protect( Rf_duplicate( obj_r ) );

  Rf_unprotect(1);
  return duplicate_r;
}

// I have 0 idea how to use the dimgets() function
extern "C"
SEXP cuR_object_recut( SEXP obj_r, SEXP dims_r ){
  Rf_setAttrib( obj_r, Rf_install("dim"), dims_r );

  return R_NilValue;
}

