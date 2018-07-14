# .Calls: src/streams.cpp

# CUDA and thread streams ====
# This class is a variation of the fusion context, but alas not inheriting from
# it
stream <- R6Class(
  "cuR.stream",
  inherit = .alert.send,
  public = list(
    initialize = function( deployed = NULL, device = cuda.device.default.get() ){
      self$device <- device
      self$deploy

      if( is.null( deployed ) ){
        return()
      }

      self$deploy( deployed )
    },

    deploy = function( level ){
      if( is.null( level ) ){
        stop( "Invalid deployment target level" )
      }

      if( !( level %in% c( 1L, 3L ) ) ){
        stop( "Invalid deployment target level" )
      }

      if( level == 1L ){
        private$.deploy.L1( expression(
          list( queue  = .Call( "cuR_stream_queue_create", 1L, FALSE ) )
        ) )
      }else{
        private$.deploy.L3( expression(
          list( queue  = .Call( "cuR_stream_queue_create", 1L, TRUE ) )
        ) )
      }

      invisible( TRUE )
    },

    destroy = function(){
      private$.destroy( expression( {
        .Call( "cuR_stream_queue_destroy", private$.ptrs$queue )
      } ) )

      invisible( TRUE )
    },

    sync = function(){
      if( is.null( private$.ptrs ) ){
        stop( "Stream is destroyed" )
      }

      .Call( "cuR_stream_queue_sync", private$.ptrs$queue )

      invisible( TRUE )
    }
  )
)
