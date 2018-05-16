alert.recv <- R6Class(
  "alert.recv",
  public = list(
    alert = function(){
      stop( "Alert not implemented" )
    }
  )
)

alert.send <- R6Class(
  "alert.send",
  public = list(
    listener.add = function( obj ){
      listener <- check.alertable( obj )

      private$.listeners <- c( private$.listeners, list( listener ) )

      invisible( self )
    },

    listener.remove = function( obj ){
      match <- sapply( private$.listeners, function( listener ){
        identical( .Internal( inspect( obj ) ),
                   .Internal( inspect( listener ) ) )
      })

      if( !any( match ) ){
        stop( "Object not amongst listeners" )
      }

      private$.listeners <- private$.listeners[ !which( match ) ]

      invisible( self )
    }
  ),

  private = list(
    .listeners = list(),

    .alert = function(){
      lapply( private$.listeners, function( listener ){
        listener$alert()
      })
    }
  )
)
