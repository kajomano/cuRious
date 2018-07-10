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

# CUDA and thread streams ====
# This class is a variation of the fusion context, but alas not inheriting from
# it
stream <- R6Class(
  "cuR.stream",
  inherit = .alert.send,
  public = list(
    initialize = function( deployed = 3L, device = cuda.device.default.get() ){
      self$device <- device

      if( is.null( deployed ) ){
        return()
      }else if( deployed == 1L ){
        self$deploy.L1()
      }else if( deployed == 3L ){
        self$deploy.L3()
      }else{
        stop( "Invalid deploy target level" )
      }
    },

    deploy.L1 = function(){
      private$.deploy.L1( expression(
        list( queue  = .Call( "cuR_stream_queue_create" ) )
      ) )

      invisible( TRUE )
    },

    deploy.L3 = function(){
      private$.deploy.L3( expression(
        list( stream = .Call( "cuR_cuda_stream_create" ),
              queue  = .Call( "cuR_stream_queue_create" ) )
      ) )

      invisible( TRUE )
    },

    destroy = function(){
      private$.destroy( expression( {
        .Call( "cuR_stream_queue_destroy", private$.ptrs$queue )

        if( !is.null( private$.ptrs$stream ) ){
          .Call( "cuR_cuda_stream_destroy", private$.ptrs$stream )
        }
      } ) )

      invisible( TRUE )
    },

    sync = function(){
      if( is.null( private$.ptrs ) ){
        stop( "Stream is destroyed" )
      }

      .Call( "cuR_stream_queue_sync", private$.ptrs$queue )

      if( !is.null( private$.ptrs$stream ) ){
        .Call( "cuR_cuda_stream_sync", private$.ptrs$stream )
      }

      invisible( TRUE )
    }
  )
)
