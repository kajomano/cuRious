# .Calls: src/streams.cpp

# CUDA stream class ====
cuda.stream <- R6Class(
  "cuda.stream",
  public = list(
    initialize = function(){
      private$.stream <- .Call( "cuR_create_cuda_stream" )
    }
  ),

  private = list(
    .stream = NULL,

    check.destroyed = function(){
      if( self$is.destroyed ){
        stop( "The stream is destroyed" )
      }
    }
  ),

  active = list(
    stream = function( val ){
      private$check.destroyed()
      if( missing( val ) ) return( private$.stream )
    },

    is.destroyed = function( val ){
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

