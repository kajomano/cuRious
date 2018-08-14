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

.max.array.rank <- 2L

is.dims <- function( dims ){
  if( !is.numeric( dims ) || length( dims ) != .max.array.rank ){
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

is.span <- function( span ){
  if( !is.list( span ) || length( span ) < 1L || length( span ) > .max.array.rank ){
    return( FALSE )
  }

  for( range in span ){
    if( !is.null( range ) ){
      if( !is.numeric( range ) || length( range ) != 2L ){
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

check.span <- function( span ){
  if( !is.span( span ) ) stop( "Invalid span" )
  invisible( lapply( span, function( range ){
    if( !is.null( range ) ){
      as.integer( range )
    }else{
      NULL
    }
  }))
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
