# .Calls: src/streams.cpp

cuda.streams.sync <- function(){
  if( is.null( .Call( "cuR_sync_device" ) ) ) stop( "Streams could not be synced" )
}

# CUDA stream class ====
cuda.stream <- R6Class(
  "cuda.stream",
  public = list(
    create = function(){
      private$stream <- .Call( "cuR_create_cuda_stream" )
      # Check for errors
      if( is.null( private$stream ) ) stop( "The CUDA stream could not be created" )
    },
    destroy = function(){
      ret <- .Call( "cuR_destroy_cuda_stream", private$stream )
      # Check for errors
      if( is.null( ret ) ) stop( "The CUDA stream could not be destroyed" )
      private$stream <- NULL
    }
  ),
  private = list(
    stream = NULL
  ),
  active = list(
    get.stream = function( val ){
      if( missing(val) ){
        if( self$is.created ) stop( "The CUDA stream is not yet created" )
        private$stream
      }
    },
    is.created = function(){
      is.null( private$stream )
    }
  )
)
