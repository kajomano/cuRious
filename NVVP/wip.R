# This script is for profiling the parallel processing calls with NVVP
# as shown in Samples/005_sample_async_proc.R
library( cuRious )

n     <- 1000
mat   <- matrix( as.double(1:(n*n)), ncol = n )

in.mat.list <- lapply( 1:10, function( ... ){
  mat
})

tens.trans <- tensor$new( t(diag(1, n, n)) )
tens.trans$dive()

tens.in <- tensor$new( mat )
tens.in$create.stage()
tens.in$dive()

tens.out <- tensor$new( mat )
tens.out$create.stage()
tens.out$dive()

sync.process <- function(){
  out.mat.list <<- lapply( in.mat.list, function( mat ){
    tens.in$push( mat )
    cublas.sgemm( handle, tens.in, tens.trans, tens.out, 1, 0 )
    tens.out$pull()
  })
}

in.stream <- cuda.stream$new()
in.stream$activate()

cublas.stream <- cuda.stream$new()
cublas.stream$activate()

out.stream <- cuda.stream$new()
out.stream$activate()

tens.list <- lapply( 1:4, function(...){
  tens <- tensor$new( mat )
  tens$create.stage()
  tens$dive()
  tens
})

handle <- cublas.handle$new()
handle$activate()

out.mat.list.async <- list()

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
    out.mat.list.async[[i-2]] <<- tens.pull$pull.proc()

    cuda.stream.sync.all()
  }

  cublas.sgemm( handle, tens.list[[4]], tens.trans, tens.list[[1]], 1, 0, stream = cublas.stream )
  tens.list[[2]]$pull.prefetch.async( out.stream )
  cuda.stream.sync( out.stream )
  out.mat.list.async[[9]] <<- tens.pull$pull.proc()

  tens.list[[1]]$pull.prefetch.async( out.stream )
  cuda.stream.sync( out.stream )
  out.mat.list.async[[10]] <<- tens.list[[1]]$pull.proc()
}

# sync.process()
async.process()

print("Finished")
