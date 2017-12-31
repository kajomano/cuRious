clean.global <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  gc()
}
