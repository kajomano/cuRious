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

cuda.device.sync <- function( device = cuda.device.default.get() ){
  device <- check.device( device )
  .cuda.device.set( device )
  .Call( "cuR_device_sync" )

  invisible( TRUE )
}

# CUDA streams ====
stream <- R6Class(
  "cuR.stream",
  inherit = .alert.send,
  public = list(
    initialize = function( deployed = TRUE, device = cuda.device.default.get() ){
      self$device <- device

      if( deployed ){
        self$deploy()
      }
    },

    deploy = function(){
      if( is.null( private$.ptrs ) ){
        private$.deploy( expression(
          list( stream = .Call( "cuR_cuda_stream_create" ) )
        ) )
      }

      invisible( TRUE )
    },

    destroy = function(){
      if( !is.null( private$.ptrs ) ){
        private$.destroy( expression(
          .Call( "cuR_cuda_stream_destroy", private$.ptrs$stream )
        ) )
      }

      invisible( TRUE )
    },

    sync = function(){
      if( self$is.deployed ){
        .Call( "cuR_cuda_stream_sync", private$.ptrs$stream )

        invisible( TRUE )
      }else{
        stop( "Stream is destroyed" )
      }
    }
  )
)
