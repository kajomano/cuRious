# .Calls: src/cublas.cpp
#
# A cublas context handle needs to be created and passed to each cublas call.
# The R finalizer is writte so that upon removal of the handle object, the
# context will be also destroyed. Keeping a single handle through multiple
# cublas call, or even thorugh the whole session is advisable.

# cuBLAS handle class ====
cublas.handle <- R6Class(
  "cublas.handle",
  public = list(
    initialize = function(){
      private$handle <- .Call( "cuR_create_cublas_handle" )
      # Check for errors
      if( is.null( private$handle ) ) stop( "cuBLAS handle could not be created" )
    }
  ),
  private = list(
    handle = NULL
  ),
  active = list(
    get.handle = function( val ){
      if( missing(val) ) return( private$handle )
    }
  )
)
