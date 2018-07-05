library( cuRious )
library( microbenchmark )
source( "./Tests/test_utils.R" )

verbose <- TRUE

test.files <- dir( "./Tests", "^[0-9]", full.names = TRUE )
test.files <- test.files[ test.files != "./Tests/000_all.R" ]

lapply( test.files, function( test.file ){
  print( test.file )

  # ----------------------------------------------------------------------------
  mult <- 1
  source( test.file, local = TRUE )

  # Test call overhead
  bench.unit <- microbenchmark( unit$run() )
  call.oh <- min( bench.unit$time )

  if( call.oh > 50000 ){
    stop( paste0( "Excessive call overhead: ", call.oh ) )
  }

  # Test L0 call
  L0$run()

  # Test L3 call - sync
  L3$run()

  # Test equality
  if( !test( verbose ) ){
    stop( "Non-identical results across levels" )
  }

  # Test L3 call - async
  # ----------------------------------------------------------------------------
  mult <- 100
  source( test.files[[1]], local = TRUE )

  stream$deploy()
  context$deploy()

  bench.L3 <- microbenchmark( L3$run() )

  if( min( bench.L3$time ) > 1.2 * call.oh ){
    stop( "Suspected blocking on async calls" )
  }

  stream$sync()

  # Cleanup
  rm( list = ls() )
  gc()
})

clean()
