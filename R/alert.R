alert.recv <- R6Class(
  "alert.recv",
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
  "alert.send",
  public = list(
    listener.add = function( obj ){
      listener <- check.alertable( obj )
      private$.listeners <- c( private$.listeners, list( listener ) )
      print( private$.listeners )
      invisible( self )
    },

    listener.remove = function(){
      match <- sapply( private$.listeners, `[[`, "listener.remove" )
      private$.listeners <- private$.listeners[ !which( match ) ]
      invisible( self )
    }
  ),

  private = list(
    .listeners = list(),

    .alert = function(){
      print( "Alerting" )
      lapply( private$.listeners, function( listener ){
        listener$alert()
      })
    }
  )
)
