# .Calls: src/streams.cpp

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

cuda.device.sync <- function( device ){
  device <- check.device( device )
  .cuda.device.set( device )
  .Call( "cuR_device_sync" )

  invisible( TRUE )
}

# CUDA streams ====
cuda.stream <- R6Class(
  "cuR.cuda.stream",
  inherit = context,
  public = list(
    sync = function(){
      if( self$is.active ){
        .Call( "cuR_cuda_stream_sync", private$.ptr )

        invisible( TRUE )
      }else{
        stop( "CUDA stream is not yet activated" )
      }
    }
  ),

  private = list(
    .activate = function(){
      .Call( "cuR_cuda_stream_create" )
    },

    .deactivate = function(){
      .Call( "cuR_cuda_stream_destroy", private$.ptr )
    }
  )
)
