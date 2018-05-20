# .Calls: src/streams.cpp

# CUDA stream class ====
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

cuda.stream.sync.all <- function(){
  if( is.null( .Call( "cuR_sync_device" ) ) ){
    stop( "Streams could not be synced" )
  }
  invisible( TRUE )
}

cuda.stream.sync <- function( stream ){
  stream <- check.cuda.stream( stream )

  if( is.null( .Call( "cuR_sync_cuda_stream", stream$stream ) ) ){
    stop( "Stream could not be synced" )
  }

  invisible( TRUE )
}

