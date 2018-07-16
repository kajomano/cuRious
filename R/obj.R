# Dimension encoder
obj.dims <- function( obj ){
  obj <- check.obj( obj )

  switch(
    class( obj )[[1]],
    matrix  = c( nrow( obj ), ncol( obj ) ),
    numeric = c( 1L, length( obj ) ),
    integer = c( 1L, length( obj ) ),
    logical = c( 1L, length( obj ) )
  )
}

# Get storage type
obj.type <- function( obj ){
  obj <- check.obj( obj )

  switch(
    class( obj )[[1]],
    matrix     = {
      mode <- storage.mode( obj )
      if( mode == "double" ) mode <- "numeric"

      if( !(mode %in% types) ){
        stop("Invalid type")
      }

      names(types)[[ which( types == mode ) ]]
    },
    numeric = "n",
    integer = "i",
    logical = "l"
  )
}

# Create a placeholder object
obj.create <- function( dims, type = "n" ){
  dims <- check.dims( dims )
  type <- check.type( type )

  if( dims[[1]] == 1 ){
    vector( types[[type]], dims[[2]] )
  }else{
    matrix( vector( types[[type]], 1 ),
            nrow = dims[1],
            ncol = dims[2] )
  }
}

# Subset an object with the same rules as cuRious does
# ( Vectors normally, matrices by column )
obj.subset <- function( obj, subset = NULL ){
  obj    <- check.obj( obj )

  if( is.null( subset ) ){
    return( obj )
  }

  subset <- check.obj( subset )

  if( obj.type( subset ) != "i" || obj.dims( subset )[[1]] != 1L ){
    stop( "Invalid subset" )
  }

  if( obj.dims( obj )[[1]] == 1L ){
    obj[ subset ]
  }else{
    obj[, subset ]
  }
}
