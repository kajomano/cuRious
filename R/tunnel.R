tunnel <- R6Class(
  "tunnel",
  inherit = alertable,
  public = list(
    initialize = function( src,
                           dst,
                           src.perm = NULL,
                           dst.perm = NULL,
                           src.span = NULL,
                           dst.span = NULL,
                           stream   = NULL  ){

      private$.src <- check.tensor( src )
      private$.dst <- check.tensor( dst )

      private$.src$alert.add( self )
      private$.dst$alert.add( self )


    },

    alert = function(){
      print( "Alerted" )
    }
  ),

  private = list(
    .src = NULL,
    .dst = NULL
  )
)
