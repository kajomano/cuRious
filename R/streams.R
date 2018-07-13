# .Calls: src/streams.cpp

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
        list( queue  = .Call( "cuR_stream_queue_create", 1L, FALSE ) )
      ) )

      invisible( TRUE )
    },

    deploy.L3 = function(){
      private$.deploy.L3( expression(
        list( queue  = .Call( "cuR_stream_queue_create", 1L, TRUE ) )
      ) )

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
