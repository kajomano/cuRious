alertable <- R6Class(
  "alertable",
  public = list(
    alert = function(){
      stop( "Alert not implemented" )
    }
  )
)
