# TPipes are for transferring data between tensors. Pipes do most of the
# argument sanity checks at creation, and try to do the rest only when needed at
# runtime. Overhead reduction is key for smaller tasks.

pipe <- R6Class(
  "cuR.pipe",
  inherit = fusion,
  public = list(
    initialize = function( src,
                           dst,
                           src.perm = NULL,
                           dst.perm = NULL,
                           src.span = NULL,
                           dst.span = NULL,
                           stream   = NULL  ){
      # Sanity checks
      check.tensor( src )
      check.tensor( dst )

      if( src$type != dst$type ){
        stop( "Tensor types do not match" )
      }

      # Dim checks
      src.dims <- .tensor.dims$new( src )
      dst.dims <- .tensor.dims$new( dst )

      if( !is.null( src.perm ) ){
        src.dims$check.perm( src.perm )
      }

      if( !is.null( dst.perm ) ){
        dst.dims$check.perm( dst.perm )
      }

      if( !is.null( src.span ) ){
        src.dims$check.span( src.span )
      }

      if( !is.null( dst.span ) ){
        dst.dims$check.span( dst.span )
      }

      if( !identical( src.dims$dims, dst.dims$dims ) ){
        stop( "Dimensions do not match" )
      }

      # Assignments
      private$.eps.fix$src <- src
      private$.eps.fix$dst <- dst

      private$.params$type <- src$type
      private$.params$dims <- src.dims$dims

      private$.params$src.span.off <- src.dims$span.off
      private$.params$dst.span.off <- dst.dims$span.off

      private$.eps.opt$src.perm <- src.perm
      private$.eps.opt$dst.perm <- dst.perm

      if( !is.null( stream ) ){
        check.cuda.stream( stream )
      }

      private$.eps.opt$stream <- stream

      super$initialize()
    }
  ),

  private = list(
    .update = function(){
      # Since levels are the primary dynamically changing attribute of tensors,
      # these checks mostly concern them
      src <- private$.eps.fix$src
      dst <- private$.eps.fix$dst

      private$.params$src.ptr <- src$ptr
      private$.params$dst.ptr <- dst$ptr

      private$.params$src.level <- src$level
      private$.params$dst.level <- dst$level

      low.cross <- ( ( src$is.level( c( 1L, 2L ) ) && dst$is.level( 3L ) ) ||
                     ( src$is.level( 3L ) && dst$is.level( c( 1L, 2L ) ) ) )

      deep.transf <- ( src$is.level( 3L ) && dst$is.level( 3L ) )

      src.perm <- private$.eps.opt$src.perm
      dst.perm <- private$.eps.opt$dst.perm

      if( !is.null( src.perm ) ){
        if( low.cross ){
          stop( "Source permutation is not available between these levels" )
        }

        if( ( deep.transf && !src.perm$is.level( 3L ) ) ||
            ( !deep.transf && src.perm$is.level( 3L ) ) ){
          stop( "Source permutation tensor is not on the correct level" )
        }

        private$.params$src.perm.ptr <- src.perm$ptr
      }

      if( !is.null( dst.perm ) ){
        if( low.cross ){
          stop( "Destination permutation is not available between these levels" )
        }

        if( ( deep.transf && !dst.perm$is.level( 3L ) ) ||
            ( !deep.transf && dst.perm$is.level( 3L ) ) ){
          stop( "Destination permutation tensor is not on the correct level" )
        }

        private$.params$dst.perm.ptr <- dst.perm$ptr
      }

      if( !is.null( private$.eps.opt$stream ) ){
        if( !is.null( private$.eps.opt$stream$stream ) ){
          if( src$is.level( c( 0L, 1L ) ) || dst$is.level( c( 0L, 1L ) ) ){
            warning( "An active stream is given to a synchronous transfer" )
          }

          private$.params$stream.ptr <- private$.eps.opt$stream$stream
        }
      }

      # Multi or single-step transfer
      if( ( src$is.level( 0L ) && dst$is.level( 3L ) ) ||
          ( src$is.level( 3L ) && dst$is.level( 0L ) ) ){
        private$.fun <- .transfer.ptr.multi
      }else{
        private$.fun <- .transfer.ptr
      }

      super$.update()
    }
  )
)
