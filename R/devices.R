# .Calls: src/devices.cpp

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

# CUDA devices ====
cuda.device.count <- function(){
  .Call( "cuR_device_count" )
}

.cuda.device.get <- function(){
  .Call( "cuR_device_get" )
}

.cuda.device.set <- function( device ){
  if( .cuRious.env$cuda.device.current == device ){
    return()
  }

  .Call( "cuR_device_set", device )
  assign( "cuda.device.current", device, envir = .cuRious.env )
}

cuda.device.default.get <- function(){
  .cuRious.env$cuda.device.default
}

cuda.device.default.set <- function( device ){
  device <- check.device( device )
  .cuda.device.set( device )
  assign( "cuda.device.default", device, envir = .cuRious.env )
}
