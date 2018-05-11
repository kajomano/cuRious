# Clean global env and restart session
clean <- function(){
  rm( list = ls( globalenv() ), pos = globalenv() )
  # .rs.restartR()
}
