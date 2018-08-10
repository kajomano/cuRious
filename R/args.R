# Argument sanity checks ====
is.level <- function( level ){
  if( !is.numeric( level ) || length( level ) != 1 ){
    return( FALSE )
  }

  if( level < 0 || level > 3 || as.logical( level %% 1 ) ){
    return( FALSE )
  }

  TRUE
}

check.level <- function( level ){
  if( !is.level( level ) ) stop( "Invalid level" )
  invisible( as.integer( level ) )
}

is.dims <- function( dims ){
  if( !is.numeric( dims ) || length( dims ) != 2 ){
    return( FALSE )
  }

  if( any( dims < 1 ) || any( as.logical( dims %% 1 ) ) ){
    return( FALSE )
  }

  TRUE
}

check.dims <- function( dims ){
  if( !is.dims( dims ) ) stop( "Invalid dims" )
  invisible( as.integer( dims ) )
}

types <- c( n = "numeric", i = "integer", l = "logical" )

is.type <- function( type ){
  !is.na( pmatch( type, types )[[1]] )
}

check.type <- function( type ){
  if( !is.type( type ) ) stop( "Invalid type" )
  type <- names( match.arg( type, types, T ) )[[1]]
  invisible( type )
}

is.device <- function( device ){
  device.count <- cuda.device.count()

  if( device.count == -1 ){
    lower.bound = -1
  }else{
    lower.bound = 0
  }

  if( !is.numeric( device ) || length( device ) != 1 ){
    return( FALSE )
  }

  if( device < lower.bound || device >= device.count || as.logical( device %% 1 ) ){
    return( FALSE )
  }

  TRUE
}

check.device <- function( device ){
  if( !is.device( device ) ) stop( "Invalid device" )
  invisible( as.integer( device ) )
}
