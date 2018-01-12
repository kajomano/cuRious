# This script shows how to use streams and asynchronous tensor push/pull
# operations combined with asychronous kernel launches. This technique is
# utilized to great effect to hide the latency of data transfers between
# the host and the device. This is a longer sample script, so be prepared!
# To keep the console output clean, it is advised to build the package without
# debug prints. To do this, comment out the definition #define DEBUG_PRINTS 1
# in the src/debug.h file.
library( cuRious )
library( microbenchmark )

n <- 1000
mat.dummy <- matrix( 0, ncol = n, nrow = n )

tens.in <- tensor$new( mat.dummy )
tens.in$create.stage()
tens.in$dive()

tens.out <- tensor$new( mat.dummy )
tens.out$create.stage()
tens.out$dive()

tens.trans <- tensor$new( t(diag(1, n, n)) )
tens.trans$dive()

in.mat.list <- lapply( 1:10, function( ... ){
  matrix( rnorm(n*n), ncol = n )
})

out.mat.list <- lapply( 1:10, function( ... ){
  matrix( 0, ncol = n, nrow = n )
})

handle <- cublas.handle$new()
handle$activate()

sync.process <- function(){
  for( i in 1:10 ){
    tens.in$push( in.mat.list[[i]] )
    cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 )
    tens.out$pull( out.mat.list[[i]] )
  }
}

in.stream <- cuda.stream$new()
in.stream$activate()

out.stream <- cuda.stream$new()
out.stream$activate()

async.transfer <- function(){
  for( i in 1:10 ){
    tens.out$pull.prefetch.async( out.stream )
    tens.in$push.preproc( mat.dummy )
    tens.in$push.fetch.async( in.stream )
    cuda.stream.sync( out.stream )
    tens.out$pull.proc( mat.dummy )
    cuda.stream.sync( in.stream )
  }
}

sync.transfer <- function(){
  for( i in 1:10 ){
    tens.in$push( mat.dummy )
    tens.out$pull( mat.dummy )
  }
}

cublas.stream <- cuda.stream$new()
cublas.stream$activate()

tens.list <- lapply( 1:4, function(...){
  tens <- tensor$new( mat.dummy )
  tens$create.stage()
  tens$dive()
  tens
})

async.process <- function(){
  tens.list[[1]]$push.preproc( in.mat.list[[1]] )
  tens.list[[1]]$push.fetch.async( in.stream )
  cuda.stream.sync.all()

  cublas.sgemm( handle, tens.list[[1]], tens.trans, tens.list[[2]], 1, 0, stream = cublas.stream  )
  tens.list[[4]]$push.preproc( in.mat.list[[2]] )
  tens.list[[4]]$push.fetch.async( in.stream )
  cuda.stream.sync.all()

  for( i in 3:10 ){
    tens.push     <- tens.list[[4 - ((i+2) %% 4)]]
    tens.gemm.in  <- tens.list[[4 - ((i+1) %% 4)]]
    tens.gemm.out <- tens.list[[4 - ((i)   %% 4)]]
    tens.pull     <- tens.list[[4 - ((i+3) %% 4)]]

    cublas.sgemm( handle, tens.gemm.in, tens.trans, tens.gemm.out, 1, 0, stream = cublas.stream )

    tens.pull$pull.prefetch.async( out.stream )
    tens.push$push.preproc( in.mat.list[[i]] )
    tens.push$push.fetch.async( in.stream )
    cuda.stream.sync( out.stream )
    tens.pull$pull.proc( out.mat.list[[i-2]] )

    cuda.stream.sync.all()
  }

  cublas.sgemm( handle, tens.list[[4]], tens.trans, tens.list[[1]], 1, 0, stream = cublas.stream )
  tens.list[[2]]$pull.prefetch.async( out.stream )
  cuda.stream.sync( out.stream )
  tens.pull$pull.proc( out.mat.list[[9]] )

  tens.list[[1]]$pull.prefetch.async( out.stream )
  cuda.stream.sync( out.stream )
  tens.list[[1]]$pull.proc( out.mat.list[[10]] )

  TRUE
}

# Comment out the one you want to see
# sync.transfer()
# async.transfer()
# sync.process()
async.process()

print("Finished")
