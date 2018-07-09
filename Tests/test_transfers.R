source( "./Tests/test_utils.R" )

verbose <- TRUE

stream <- cuda.stream$new()

for( type in types ){
  mat.cont <- switch(
    type,
    numeric = as.numeric( 1:(1000*1000) ),
    integer = as.integer( 1:(1000*1000) ),
    logical = as.logical( 1:(1000*1000) )
  )

  mat <- matrix( mat.cont, 1000, 1000 )

  for( src.level in 0:3 ){
    for( dst.level in 0:3 ){
      print( paste0( type, " ", src.level, " ", dst.level ) )

      src <- tensor$new( mat, src.level )
      dst <- tensor$new( mat, dst.level, copy = FALSE )

      if( src.level == 3L && dst.level == 3L ){
        perm <- tensor$new( 1:1000, 3L )
      }else{
        perm <- tensor$new( 1:1000, 0L )
      }

      pip.sync  <- pipe$new( src, dst )
      pip.async <- pipe$new( src, dst, stream = stream )

      pip.sync.perm  <- pipe$new( src, dst, perm, perm )
      pip.async.perm <- pipe$new( src, dst, perm, perm, stream = stream )

      pip.sync$run()
      if( !test.thr.equality( dst$pull(), mat ) ){
        stop( "Failed check: sync" )
      }

      dst$clear()

      pip.async$run()
      stream$sync()
      if( !test.thr.equality( dst$pull(), mat ) ){
        stop( "Failed check: async" )
      }

      dst$clear()

      pip.sync.perm$run()
      if( !test.thr.equality( dst$pull(), mat ) ){
        stop( "Failed check: sync perm" )
      }

      dst$clear()

      pip.async.perm$run()
      stream$sync()
      if( !test.thr.equality( dst$pull(), mat ) ){
        stop( "Failed check: async perm" )
      }

      if( verbose ){
        if( type == "numeric" ){
          bench.sync  <- microbenchmark( pip.sync$run(), times = 100 )
          bench.async <- microbenchmark( pip.async$run(), times = 100 )
          synced <- function(){
            pip.async$run()
            stream$sync()
          }
          bench.synced <- microbenchmark( synced(), times = 100 )

          bench.sync.perm  <- microbenchmark( pip.sync.perm$run(), times = 100 )
          bench.async.perm <- microbenchmark( pip.async.perm$run(), times = 100 )
          synced.perm <- function(){
            pip.async.perm$run()
            stream$sync()
          }
          bench.synced.perm <- microbenchmark( synced.perm(), times = 100 )

          print( paste0( "sync: ", min( bench.sync$time ) / 1000, " us" ) )
          print( paste0( "async: ", min( bench.async$time ) / 1000, " us" ) )
          print( paste0( "synced: ", min( bench.synced$time ) / 1000, " us" ) )

          print( paste0( "sync.perm: ", min( bench.sync.perm$time ) / 1000, " us" ) )
          print( paste0( "async.perm: ", min( bench.async.perm$time ) / 1000, " us" ) )
          print( paste0( "synced.perm: ", min( bench.synced.perm$time ) / 1000, " us" ) )
        }
      }
    }
  }
}
