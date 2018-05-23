# .Calls: src/streams.cpp

# CUDA devices ====
cuda.device.count <- function(){
  ret <- .Call( "cuR_device_count" )
  if( is.null( ret ) ){
    stop( "Failed to get device count" )
  }
  ret
}

cuda.device.get <- function( ){
  ret <- .Call( "cuR_get_device" )
  if( is.null( ret ) ){
    stop( "Failed to get current device" )
  }
  ret
}

cuda.device.set <- function( device ){
  device <- check.device( device )

  if( is.null( .Call( "cuR_set_device", device ) ) ){
    stop( "Failed to set current device" )
  }
  invisible( TRUE )
}

cuda.device.sync <- function(){
  if( is.null( .Call( "cuR_sync_device" ) ) ){
    stop( "Streams could not be synced" )
  }
  invisible( TRUE )
}

# CUDA streams ====
cuda.stream <- R6Class(
  "cuR.cuda.stream",
  inherit = alert.send,
  public = list(
    initialize = function( active = T ){
      if( active ){
        self$activate()
      }
    },

    activate = function(){
      if( is.null( private$.stream ) ){
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
    .stream = NULL
  ),

  active = list(
    stream = function( val ){
      if( missing( val ) ) return( private$.stream )
    },

    is.active = function( val ){
      if( missing( val ) ) return( is.null( private$.stream ) )
    }
  )
)
