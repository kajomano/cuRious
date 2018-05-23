# .Calls: src/streams.cpp

# CUDA devices ====
cuda.device.count <- function(){
  ret <- .Call( "cuR_device_count" )
  if( is.null( ret ) ){
    stop( "Failed to get device count" )
  }
  ret
}

.cuda.device.get <- function( ){
  device <- .Call( "cuR_get_device" )
  if( is.null( device ) ){
    stop( "Failed to get current device" )
  }

  device
}

cuda.device.get <- function(){
  .cuda.device.get()
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

cuda.device.set <- function(){
  .cuda.device.set()
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
  inherit = alert.send,
  public = list(
    initialize = function( device = 0L ){
      if( !is.null( device ) ){
        private$.device <- check.device( device )
        self$activate()
      }else{
        private$.device <- 0L
      }
    },

    activate = function(){
      if( is.null( private$.stream ) ){
        .cuda.device.set( private$.device )
        private$.stream <- .Call( "cuR_create_cuda_stream" )
        private$.alert()
      }else{
        warning( "CUDA stream is already activated" )
      }

      invisible( self )
    },

    deactivate = function(){
      if( !is.null( private$.stream ) ){
        .Call( "cuR_destroy_cuda_stream", private$.stream )
        private$.stream <- NULL
        private$.alert()
      }else{
        warning( "CUDA stream is not yet activated" )
      }

      invisible( self )
    },

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
    .stream = NULL,
    .device = NULL
  ),

  active = list(
    stream = function( val ){
      if( missing( val ) ) return( private$.stream )
    },

    device = function( device ){
      if( missing( device ) ){
        return( private$.device )
      }else{
        device <- check.device( device )

        if( private$.device == device ){
          return()
        }

        if( self$is.active ){
          stop( "Cannot change device: active stream" )
        }

        private$.device <- device
      }
    },

    is.active = function( val ){
      if( missing( val ) ) return( is.null( private$.stream ) )
    }
  )
)
