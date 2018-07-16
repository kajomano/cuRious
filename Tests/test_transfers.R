source( "./Tests/test_utils.R" )

verbose <- TRUE

threads <- 4

cols    <- 1000
rows    <- 1000

stream        <- cuRious::stream$new( deployed = 3 )
context.sync  <- cuRious::pipe.context$new( threads, deployed = 3 )
context.async <- cuRious::pipe.context$new( threads, stream, deployed = 3 )

for( type in types ){
  mat.cont <- switch(
    type,
    numeric = as.numeric( 1:(cols*rows) ),
    integer = as.integer( 1:(cols*rows) ),
    logical = as.logical( 1:(cols*rows) )
  )

  mat <- matrix( mat.cont, rows, cols )

  for( src.level in 0:3 ){
    for( dst.level in 0:3 ){
  # for( src.level in 0 ){
  #   for( dst.level in 0 ){
      if( ( src.level == 0L && dst.level == 3L ) ||
          ( src.level == 3L && dst.level == 0L ) ){
        next
      }

      print( paste0( type, " ", src.level, " ", dst.level ) )

      src <- tensor$new( mat, src.level )
      dst <- tensor$new( mat, dst.level, copy = FALSE )

      perm.src <- NULL
      perm.dst <- NULL

      if( src.level == 3L && dst.level == 3L ){
        perm.src <- cuRious::tensor$new( 1:cols, 3L )
        perm.dst <- cuRious::tensor$new( 1:cols, 3L )
      }else{
        perm.src <- cuRious::tensor$new( 1:cols, 0L )
        perm.dst <- cuRious::tensor$new( 1:cols, 0L )
      }

      pip.sync       <- cuRious::pipe$new( src, dst, context = context.sync )
      pip.async      <- cuRious::pipe$new( src, dst, context = context.async )

      pip.sync.perm  <- cuRious::pipe$new( src, dst, perm.src, perm.dst, context = context.sync )
      pip.async.perm <- cuRious::pipe$new( src, dst, perm.src, perm.dst, context = context.async )

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
