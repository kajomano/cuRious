# .Calls: src/streams.cpp

cuda.stream.sync.all <- function(){
  if( is.null( .Call( "cuR_sync_device" ) ) ){
    stop( "Streams could not be synced" )
  }
  invisible( TRUE )
}

cuda.stream.sync <- function( stream ){
  check.cuda.stream.active( stream )
  if( is.null( .Call( "cuR_sync_cuda_stream", stream$get.stream ) ) ){
    stop( "Stream could not be synced" )
  }
  invisible( TRUE )
}

# CUDA stream class ====
cuda.stream <- R6Class(
  "cuda.stream",
  public = list(
    activate = function(){
      if( self$is.active ){
        stop( "The CUDA stream is already active" )
      }
      private$stream <- .Call( "cuR_activate_cuda_stream" )

      if( is.null( private$stream ) ){
        stop( "The CUDA stream could not be activated" )
      }

      invisible( TRUE )
    },
    deactivate = function(){
      if( !self$is.active ){
        stop( "The CUDA stream has not yet been activated" )
      }

      .Call( "cuR_deactivate_cuda_stream", private$stream )
      private$stream <- NULL

      invisible( TRUE )
    }
  ),

  private = list(
    stream = NULL
  ),

  active = list(
    get.stream = function( val ){
      if( missing(val) ) return( private$stream )
    },
    is.active = function(){
      !is.null( private$stream )
    }
  )
)

is.cuda.stream <- function( ... ){
  objs <- list( ... )
  sapply( objs, function( obj ){
    "cuda.stream" %in% class( obj )
  })
}

check.cuda.stream <- function( ... ){
  if( !all( is.cuda.stream( ... ) ) ){
    stop( "Not all objects are CUDA streams" )
  }
}

is.cuda.stream.active <- function( ... ){
  check.cuda.stream( ... )

  streams <- list( ... )
  sapply( streams, function( stream ){
    stream$is.active
  })
}

check.cuda.stream.active <- function( ... ){
  if( !all( is.cuda.stream.active( ... ) ) ){
    stop( "Not all CUDA streams are active" )
  }
}
