# .Calls: src/algebra.c

vect.add <- function( obj.l, obj.r, obj.res ){
  if( !is.struct(obj.l) || !is.struct(obj.r) || !is.struct(obj.res) ){
    stop( "Not all objects are mathematical structures" )
  }

  if( !is.vect(obj.l) || !is.vect(obj.r) || !is.vect(obj.res) ){
    stop( "Not all objects are vectors" )
  }

  if( obj.l$l != obj.r$l || obj.l$l != obj.res$l ){
    stop( "Not all vectors are the same length" )
  }

  if( !is.under(obj.l) || !is.under(obj.r) || !is.under(obj.res) ){
    stop( "Not all vectors are under" )
  }

  .Call()
}
