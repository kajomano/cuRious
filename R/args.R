# Argument sanity checks

is.obj <- function( obj ){
  switch(
    class( obj )[[1]],
    matrix     = TRUE,
    numeric    = TRUE,
    integer    = TRUE,
    logical    = TRUE,
    FALSE
  )
}

check.obj <- function( obj ){
  if( !is.obj( obj ) ) stop( "Unsupported object" )
  invisible( obj )
}

is.tensor <- function( tensor ){
  "cuR.tensor" %in% class( tensor )
}

check.tensor <- function( tensor ){
  if( !is.tensor( tensor ) ) stop( "Not a tensor" )
  invisible( tensor )
}

types <- c( n = "numeric", i = "integer", l = "logical" )

is.type <- function( type ){
  !is.na( pmatch( type, types ) )
}

check.type <- function( type ){
  if( !is.type( type ) ) stop( "Invalid type" )
  type <- names( types )[[ pmatch( type, types ) ]]
  invisible( type )
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

is.cuda.stream <- function( stream ){
  "cuR.cuda.stream" %in% class( stream )
}

check.cuda.stream <- function( stream ){
  if( !is.cuda.stream( stream ) ) stop( "Invalid CUDA stream" )
  invisible( stream )
}

is.cublas.handle <- function( handle ){
  "cuR.cublas.handle" %in% class( handle )
}

check.cublas.handle <- function( handle ){
  if( !is.cublas.handle( handle ) ) stop( "Invalid cuBLAS handle" )
  invisible( handle )
}

is.alertable <- function( alertable ){
  "cuR.alert.receiver" %in% class( alertable )
}

check.alertable <- function( alertable ){
  if( !is.alertable( alertable ) ) stop( "Invalid alertable" )
  invisible( alertable )
}

is.alerting <- function( alerting ){
  "cuR.alert.sender" %in% class( alerting )
}

check.alerting <- function( alerting ){
  if( !is.alerting( alerting ) ) stop( "Invalid alerting" )
  invisible( alerting )
}

is.device <- function( device ){
  if( !is.numeric( device ) || length( device ) != 1 ){
    return( FALSE )
  }

  if( device < 0 || device >= cuda.device.count() || as.logical( device %% 1 ) ){
    return( FALSE )
  }

  TRUE
}

check.device <- function( device ){
  if( !is.device( device ) ) stop( "Invalid device" )
  invisible( as.integer( device ) )
}
