source( "./Tests/test_utils.R" )

verbose <- 2

test.files <- dir( "./Tests", "^[0-9]", full.names = TRUE )
test.files <- test.files[ test.files != "./Tests/000_all.R" ]

lapply( test.files, function( test.file ){
  print( test.file )

  # Test results
  # ----------------------------------------------------------------------------
  mult <- 1
  source( test.file, local = TRUE )

  # Test L0 call
  L0$run()

  context$level <- 3L

  # Test L3 call - sync
  L3$run()

  # Test equality
  if( !test( verbose > 1 ) ){
    stop( "Non-identical results across levels" )
  }

  stream$level <- 3L

  # Test L3 call - async
  clear()
  L3$run()
  stream$sync()

  # Test equality
  if( !test( verbose > 1 ) ){
    stop( "Non-identical results across levels (async)" )
  }

  # Benchmark
  # ----------------------------------------------------------------------------
  if( verbose > 0 ){
    mult <- 100
    source( test.file, local = TRUE )

    context$level <- 3L

    bench.sync  <- microbenchmark( L3$run(), times = 100 )

    stream$level <- 3L

    bench.async <- microbenchmark( L3$run(), times = 100 )

    synced <- function(){
      L3$run()
      stream$sync()
    }
    bench.synced <- microbenchmark( synced(), times = 100 )

    print( paste0( "sync: ", min( bench.sync$time ) / 1000, " us" ) )
    print( paste0( "async: ", min( bench.async$time ) / 1000, " us" ) )
    print( paste0( "synced: ", min( bench.synced$time ) / 1000, " us" ) )
  }

  # Cleanup
  # clean()
})
