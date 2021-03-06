# .Calls: src/streams.cpp

# CUDA and thread streams ====
# This class is a variation of the fusion context, but alas not inheriting from
# it
stream <- R6Class(
  "cuR.stream",
  inherit = .alert.send,
  public = list(
    sync = function(){
      self$check.destroyed()

      if( private$.level ){
        .Call( "cuR_stream_queue_sync", private$.ptrs$queue )
      }

      invisible( self )
    }
  ),

  private = list(
    .deploy.L0 = function(){
      list( queue = NULL )
    },

    .deploy.L1 = function(){
      list( queue  = .Call( "cuR_stream_queue_create", 1L, FALSE ) )
    },

    .deploy.L3 = function(){
      list( queue  = .Call( "cuR_stream_queue_create", 1L, TRUE ) )
    },

    .destroy.L0 = function(){
      return()
    },

    .destroy.L1 = function(){
      .Call( "cuR_stream_queue_destroy", private$.ptrs$queue )
    },

    .destroy.L3 = function(){
      .Call( "cuR_stream_queue_destroy", private$.ptrs$queue )
    }
  )
)
