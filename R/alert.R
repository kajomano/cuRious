alert.recv <- R6Class(
  "cuR.alert.receiver",
  public = list(
    alert = function(){
      stop( "Alert not implemented" )
    }
  ),

  private = list(
    .listener.remove = FALSE
  ),

  active = list(
    listener.remove = function( val ){
      if( missing( val ) ) return( private$.listener.remove )
    }
  )
)

alert.send <- R6Class(
  "cuR.alert.sender",
  public = list(
    listener.add = function( obj ){
      listener <- check.alertable( obj )
      private$.listeners <- c( private$.listeners, list( listener ) )

      invisible( self )
    },

    listener.remove = function(){
      match <- sapply( private$.listeners, `[[`, "listener.remove" )
      private$.listeners <- private$.listeners[ !match ]
      invisible( self )
    }
  ),

  private = list(
    .listeners = list(),

    .alert = function(){
      lapply( private$.listeners, function( listener ){
        listener$alert()
      })

      invisible( TRUE )
    }
  )
)
