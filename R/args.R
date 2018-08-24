# Argument sanity checks ====
is.level <- function( level ){
  if( !is.numeric( level ) ||
      length( level ) != 1 ){
    return( FALSE )
  }

  if( level < 0 ||
      level > 3 ||
      as.logical( level %% 1 ) ){
    return( FALSE )
  }

  TRUE
}

check.level <- function( level ){
  if( !is.level( level ) ) stop( "Invalid level" )
  invisible( as.integer( level ) )
}

.max.array.rank <- 2L

is.rank <- function( rank ){
  if( !is.numeric( rank ) ||
      length( rank ) != 1 ){
    return( FALSE )
  }

  if( rank < 1 ||
      rank > .max.array.rank ||
      as.logical( rank %% 1 ) ){
    return( FALSE )
  }

  TRUE
}

check.rank <- function( rank ){
  if( !is.rank( rank ) ) stop( "Invalid rank" )
  invisible( as.integer( rank ) )
}

is.dims <- function( dims ){
  if( !is.numeric( dims ) ||
      length( dims ) > .max.array.rank ||
      length( dims ) == 0 ){
    return( FALSE )
  }

  if( any( dims < 1 ) ||
      any( as.logical( dims %% 1 ) ) ){
    return( FALSE )
  }

  TRUE
}

check.dims <- function( dims ){
  if( !is.dims( dims ) ) stop( "Invalid dims" )
  invisible( as.integer( dims ) )
}

is.ranges <- function( ranges ){
  if( !is.list( ranges ) ||
      length( ranges ) > .max.array.rank ||
      length( ranges ) == 0 ){
    return( FALSE )
  }

  for( range in ranges ){
    if( !is.null( range ) ){
      if( !is.numeric( range ) ||
          length( range ) != 2L ){
        return( FALSE )
      }

      if( range[[1]] < 1L ||
          range[[2]] < range[[1]] ||
          any( as.logical( range %% 1 ) ) ){
        return( FALSE )
      }
    }
  }

  TRUE
}

check.ranges <- function( ranges ){
  if( !is.ranges( ranges ) ) stop( "Invalid ranges" )

  invisible( lapply( ranges, function( range ){
    if( !is.null( range ) ){
      as.integer( range )
    }else{
      NULL
    }
  }))
}

.types <- c( n = "numeric", i = "integer", l = "logical" )

is.type <- function( type ){
  !is.na( pmatch( type, .types )[[1]] )
}

check.type <- function( type ){
  if( !is.type( type ) ) stop( "Invalid type" )
  type <- names( match.arg( type, .types, T ) )[[1]]
  invisible( type )
}
