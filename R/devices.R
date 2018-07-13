# .Calls: src/devices.cpp

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
