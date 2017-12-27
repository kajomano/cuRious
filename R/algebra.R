op.fallback <- function( obj.l, obj.r, op ){
  if( is.under(obj.l) ) obj.l <- surface( obj.l )
  if( is.under(obj.r) ) obj.r <- surface( obj.r )

  op( obj.l, obj.r )
}

vect.add <- function( obj.l, obj.r, obj.res ){
  if( is.under(obj.l) && is.under(obj.r) ){
    bowser()
  }

  stop( "Not all objects are under" )

  return( op.fallback( obj.l, obj.r, `+` ) )
}
