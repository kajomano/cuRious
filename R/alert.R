alert.recv <- R6Class(
  "cuR.alert.receiver",
  public = list(
    alert = function( ... ){
      stop( "Alert not implemented" )
    },

    alert.context = function( ... ){
      stop( "Context alert not implemented" )
    },

    alert.content = function( ... ){
      stop( "Content alert not implemented" )
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
    listener.add = function( listener, name ){
      if( !( "cuR.alert.receiver" %in% class( listener ) ) ){
        stop( "Invalid listener" )
      }
      attr( listener, name ) <- name
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
      for( listener in private$.listeners ){
        listener$alert( attr( listener, name ) )
      }

      invisible( TRUE )
    },

    .alert.context = function(){
      for( listener in private$.listeners ){
        listener$alert.context( attr( listener, name ) )
      }

      invisible( TRUE )
    },

    .alert.content = function(){
      for( listener in private$.listeners ){
        listener$alert.content( attr( listener, name ) )
      }

      invisible( TRUE )
    }
  )
)
