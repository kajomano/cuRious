# .Calls: src/streams.cpp

# CUDA devices ====
cuda.device.count <- function(){
  ret <- .Call( "cuR_device_count" )
  if( is.null( ret ) ){
    stop( "Failed to get device count" )
  }
  ret
}

.cuda.device.get <- function(){
  device <- .Call( "cuR_get_device" )
  if( is.null( device ) ){
    stop( "Failed to get current device" )
  }

  device
}

.cuda.device.set <- function( device ){
  if( .cuRious.env$cuda.device.current == device ){
    return()
  }

  if( is.null( .Call( "cuR_set_device", device ) ) ){
    stop( "Failed to set current device" )
  }

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

  if( is.null( .Call( "cuR_sync_device" ) ) ){
    stop( "Device could not be synced" )
  }
  invisible( TRUE )
}

# CUDA streams ====
cuda.stream <- R6Class(
  "cuR.cuda.stream",
  inherit = context,
  public = list(
    sync = function(){
      if( self$is.active ){
        if( is.null( .Call( "cuR_sync_cuda_stream", private$.stream ) ) ){
          stop( "Stream could not be synced" )
        }

        invisible( TRUE )
      }else{
        stop( "CUDA stream is not yet activated" )
      }
    }
  ),

  private = list(
    .activate = function(){
      .Call( "cuR_create_cuda_stream" )
    },

    .deactivate = function(){
      .Call( "cuR_destroy_cuda_stream", private$.ptr )
    }
  )
)
