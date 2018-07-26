# .Calls: src/thrust.cpp
#
# A thrust allocator is required for every thrust call. Altough not every thrust
# operation requires temporary buffer allocations, whether or not one does is
# not something I want to test for each. Thrust allocators are not thread safe,
# a separate one needs to be created for use with each cuda stream and device.

# Thrust context class ====
thrust.context <- R6Class(
  "cuR.thrust.context",
  inherit = fusion.context,
  private = list(
    .deploy.L0 = function(){
      list( alloc = NULL )
    },

    .deploy.L3 = function(){
      list( alloc = .Call( "cuR_thrust_allocator_create" ) )
    },

    .destroy.L0 = function(){
      return()
    },

    .destroy.L3 = function(){
      .Call( "cuR_thrust_allocator_destroy", private$.ptrs$alloc )
    }
  )
)

# Thrust fusions ====

# pow ====
# B <- A^pow
thrust.pow <- R6Class(
  "cuR.thrust.pow",
  inherit = fusion,
  public = list(
    initialize = function( A,
                           B,
                           A.span  = NULL,
                           B.span  = NULL,
                           pow     = 2,
                           context = NULL ){
      # Sanity checks
      check.tensor( A )
      check.tensor( B )

      if( !all( c( A$type == "n", B$type == "n" ) ) ){
        stop( "All input tensors need to be numeric" )
      }

      if( !is.numeric( pow ) || !( length( pow ) == 1L ) ){
        stop( "Invalid pow parameter" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      B.dims <- .tensor.dims$new( B )

      A.dims$check.span( A.span )
      B.dims$check.span( B.span )

      if( !identical( A.dims$dims, B.dims$dims ) ){
        stop( "Tensor dimension mismatch" )
      }

      # Assignments
      private$.add.ep( A, "A" )
      private$.add.ep( B, "B", TRUE )

      private$.params$A.dims <- A.dims$dims

      private$.params$A.span.off <- A.dims$span.off
      private$.params$B.span.off <- B.dims$span.off

      private$.params$pow <- as.numeric( pow )

      super$initialize( context )
    }
  ),

  private = list(
    .call.L3 = function( A.tensor,
                         B.tensor,
                         A.dims,
                         A.span.off,
                         B.span.off,
                         pow,
                         context.alloc,
                         stream.queue  = NULL ){

      .Call( "cuR_thrust_pow",
             A.tensor,
             B.tensor,
             A.dims,
             A.span.off,
             B.span.off,
             pow,
             context.alloc,
             stream.queue )

      invisible( TRUE )
    },

    .call.L0 = function( A.tensor,
                         B.tensor,
                         A.dims,
                         A.span.off,
                         B.span.off,
                         pow,
                         context.alloc = NULL,
                         stream.queue  = NULL ){

      if( A.span.off != 1L || obj.dims( A.tensor )[[2]] != A.dims[[2]] ){
        A.tensor <- obj.subset( A.tensor, A.span.off:( A.span.off + A.dims[[2]] - 1L ) )
      }

      B.range <- NULL

      if( B.span.off != 1L || obj.dims( B.tensor )[[2]] != A.dims[[2]] ){
        B.range <- B.span.off:( B.span.off + A.dims[[2]] - 1L )
      }

      res <- A.tensor ^ pow

      if( is.null( B.range ) ){
        private$.eps.out$B$obj.unsafe <- res
      }else{
        if( A.dims[[1]] == 1L ){
          private$.eps.out$B$obj.unsafe[ B.range ] <- res
        }else{
          private$.eps.out$B$obj.unsafe[, B.range ] <- res
        }
      }

      invisible( TRUE )
    }
  )
)

# cmin pos ====
# Tells which row contains the smallest number for every column
thrust.cmin.pos <- R6Class(
  "cuR.thrust.cmin.pos",
  inherit = fusion,
  public = list(
    initialize = function( A,
                           x,
                           A.span  = NULL,
                           x.span  = NULL,
                           context = NULL  ){
      # Sanity checks
      check.tensor( A )
      check.tensor( x )

      if( A$type != "n" ){
        stop( "Input tensors A is not numeric" )
      }

      if( x$type != "i" ){
        stop( "Input tensors x is not integer" )
      }

      # Dim checks
      A.dims <- .tensor.dims$new( A )
      x.dims <- .tensor.dims$new( x )

      A.dims$check.span( A.span )
      x.dims$check.span( x.span )

      x.dims$check.vect()

      if( A.dims$dims[[2]] != x.dims$dims[[2]] ){
        stop( "Tensor dimension mismatch" )
      }

      # Assignments
      private$.add.ep( A, "A" )
      private$.add.ep( x, "x", TRUE )

      private$.params$A.dims <- A.dims$dims

      private$.params$A.span.off <- A.dims$span.off
      private$.params$x.span.off <- x.dims$span.off

      super$initialize( context )
    }
  ),

  private = list(
    .call.L3 = function( A.tensor,
                         x.tensor,
                         A.dims,
                         A.span.off,
                         x.span.off,
                         context.alloc,
                         stream.queue  = NULL ){

      .Call( "cuR_thrust_cmin_pos",
             A.tensor,
             x.tensor,
             A.dims,
             A.span.off,
             x.span.off,
             context.alloc,
             stream.queue )

      invisible( TRUE )
    },

    .call.L0 = function( A.tensor,
                         x.tensor,
                         A.dims,
                         A.span.off,
                         x.span.off,
                         context.alloc = NULL,
                         stream.queue  = NULL ){

      if( A.span.off != 1L || obj.dims( A.tensor )[[2]] != A.dims[[2]] ){
        A.tensor <- obj.subset( A.tensor, A.span.off:( A.span.off + A.dims[[2]] - 1L ) )
      }

      x.range <- NULL

      if( x.span.off != 1L || obj.dims( x.tensor )[[2]] != A.dims[[2]] ){
        x.range <- x.span.off:( x.span.off + A.dims[[2]] - 1L )
      }

      res <- apply( A.tensor, 2, which.min )

      if( is.null( x.range ) ){
        private$.eps.out$x$obj.unsafe <- res
      }else{
        private$.eps.out$x$obj.unsafe[ x.range ] <- res
      }

      invisible( TRUE )
    }
  )
)

# table ====
thrust.table <- R6Class(
  "cuR.thrust.table",
  inherit = fusion,
  public = list(
    initialize = function( x,  # Input vector tensor to be table-d
                           p,  # Output permutation for sorting to cont. blocks
                           w,  # Output weights
                           s,  # output sorted x
                           x.span  = NULL,
                           p.span  = NULL,
                           w.span  = NULL,
                           s.span  = NULL,
                           context = NULL  ){
      # Sanity checks
      check.tensor( x )
      check.tensor( p )
      check.tensor( w )
      check.tensor( s )

      if( !all( c( x$type     == "i",
                   p$type     == "i",
                   w$type     == "i",
                   s$type     == "i" ) ) ){
        stop( "All input tensors need to be integers" )
      }

      # Dim checks
      x.dims <- .tensor.dims$new( x )
      p.dims <- .tensor.dims$new( p )
      w.dims <- .tensor.dims$new( w )
      s.dims <- .tensor.dims$new( s )

      x.dims$check.span( x.span )
      p.dims$check.span( p.span )
      w.dims$check.span( w.span )
      s.dims$check.span( s.span )

      x.dims$check.vect()
      p.dims$check.vect()
      w.dims$check.vect()
      s.dims$check.vect()

      # Weight tensor dimensions are checked at runtime
      if( ( x.dims$dims[[2]] != p.dims$dims[[2]] ) ||
          ( x.dims$dims[[2]] != s.dims$dims[[2]] ) ){
        stop( "Tensor dimension mismatch" )
      }

      # Assignments
      private$.add.ep( x, "x" )
      private$.add.ep( p, "p", TRUE )
      private$.add.ep( w, "w", TRUE )
      private$.add.ep( s, "s", TRUE )

      private$.params$x.dims <- x.dims$dims
      private$.params$w.dims <- w.dims$dims

      private$.params$x.span.off <- x.dims$span.off
      private$.params$p.span.off <- p.dims$span.off
      private$.params$w.span.off <- w.dims$span.off
      private$.params$s.span.off <- s.dims$span.off

      super$initialize( context )
    }
  ),

  private = list(
    .call.L3 = function( x.tensor,
                         p.tensor,
                         w.tensor,
                         s.tensor,
                         x.dims,
                         w.dims,
                         x.span.off,
                         p.span.off,
                         w.span.off,
                         s.span.off,
                         context.alloc,
                         stream.queue  = NULL ){

      .Call( "cuR_thrust_table",
             x.tensor,
             p.tensor,
             w.tensor,
             s.tensor,
             x.dims,
             w.dims,
             x.span.off,
             p.span.off,
             w.span.off,
             s.span.off,
             context.alloc,
             stream.queue )

      invisible( TRUE )
    },

    .call.L0 = function( x.tensor,
                         p.tensor,
                         w.tensor,
                         s.tensor,
                         x.dims,
                         w.dims,
                         x.span.off,
                         p.span.off,
                         w.span.off,
                         s.span.off,
                         context.alloc,
                         stream.queue  = NULL ){

      if( x.span.off != 1L || obj.dims( x.tensor )[[2]] != x.dims[[2]] ){
        x.tensor <- x.tensor[ x.span.off:( x.span.off + x.dims[[2]] - 1L ) ]
      }

      # P
      p.res <- order( x.tensor ) + x.span.off - 1L

      p.range  <- NULL

      if( p.span.off != 1L || obj.dims( p.tensor )[[2]] != x.dims[[2]] ){
        p.range <- p.span.off:( p.span.off + x.dims[[2]] - 1L )
      }

      if( is.null( p.range ) ){
        private$.eps.out$p$obj.unsafe <- p.res
      }else{
        private$.eps.out$p$obj.unsafe[ p.range ] <- p.res
      }

      # S
      s.res <- x.tensor[ p.res ]

      s.range  <- NULL

      if( s.span.off != 1L || obj.dims( s.tensor )[[2]] != x.dims[[2]] ){
        s.range <- s.span.off:( s.span.off + x.dims[[2]] - 1L )
      }

      if( is.null( s.range ) ){
        private$.eps.out$s$obj.unsafe <- s.res
      }else{
        private$.eps.out$s$obj.unsafe[ s.range ] <- s.res
      }

      # W
      # Weight dims check
      if( s.res[[ 1  ]] < 1 ||
          s.res[[ length( s.res ) ]] > w.dims[[2]] ){
        stop( "Invalid key" )
      }

      w.res <- as.data.frame( table( s.res, dnn = "Var" ),
                              stringsAsFactors = FALSE )

      w.res <- merge( w.res,
                      data.frame(
                        Var = as.character(
                          1:w.dims[[2]]
                        )
                      ),
                      all.y = TRUE
      )$Freq

      w.res[ is.na( w.res ) ] <- 0L

      w.range  <- NULL

      if( w.span.off != 1L || obj.dims( w.tensor )[[2]] != w.dims[[2]] ){
        w.range <- w.span.off:( w.span.off + w.dims[[2]] - 1L )
      }

      if( is.null( w.range ) ){
        private$.eps.out$w$obj.unsafe <- w.res
      }else{
        private$.eps.out$w$obj.unsafe[ w.range ] <- w.res
      }

      invisible( TRUE )
    }
  )
)
