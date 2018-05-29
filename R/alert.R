alert.recv <- R6Class(
  "cuR.alert.subscriber",
  public = list(
    unsubscribe.flag = FALSE,

    alert = function( name ){
      self$check.destroyed()
      private$.add.content.changed( name )
      private$.add.context.changed( name )
    },

    alert.context = function( name ){
      self$check.destroyed()
      private$.add.context.changed( name )
    },

    alert.content = function( name ){
      self$check.destroyed()
      private$.add.content.changed( name )
    }
  ),

  private = list(
    .context.changed = NULL,
    .content.changed = NULL,

    .subscribe = function( sender, name ){
      if( !( "cuR.alert.sender" %in% class( sender ) ) ){
        stop( "Invalid sender" )
      }

      sender$subscriber.add( self, name )

      private$.add.content.changed( name )
      private$.add.context.changed( name )
    },

    .unsubscribe = function( sender, name ){
      if( !( "cuR.alert.sender" %in% class( sender ) ) ){
        stop( "Invalid sender" )
      }

      self$unsubscribe.flag <- TRUE
      sender$subscriber.remove()
      self$unsubscribe.flag <- FALSE

      private$.remove.content.changed( name )
      private$.remove.context.changed( name )
    },

    .add.content.changed = function( name ){
      name.match <- ( name == private$.content.changed )
      if( !any( name.match ) ){
        private$.content.changed <- c( private$.content.changed, name )
      }
    },

    .add.context.changed = function( name ){
      name.match <- ( name == private$.context.changed )
      if( !any( name.match ) ){
        private$.context.changed <- c( private$.context.changed, name )
      }
    },

    .remove.content.changed = function( name ){
      name.match <- ( name == private$.content.changed )
      if( any( name.match ) ){
        private$.content.changed <- private$.content.changed[ !name.match ]
      }
    },

    .remove.context.changed = function( name ){
      name.match <- ( name == private$.context.changed )
      if( any( name.match ) ){
        private$.context.changed <- private$.context.changed[ !name.match ]
      }
    }
  ),

  active = list(
    listener.remove = function( val ){
      if( missing( val ) ) return( private$.listener.remove )
    }
  )
)

alert.send <- R6Class(
  "cuR.alert.sender",
  public = list(
    subscriber.add = function( subscriber, name ){
      if( !( "cuR.alert.subscriber" %in% class( subscriber ) ) ){
        stop( "Invalid subscriber" )
      }

      subsc <- list( subscriber )
      names( subsc ) <- name
      private$.subscribers <- c( private$.subscribers, subsc )

      invisible( self )
    },

    subscriber.remove = function(){
      matched <- sapply( private$.subscribers, `[[`, "unsubscribe.flag" )
      private$.subscribers <- private$.subscribers[ !matched ]
      invisible( self )
    }
  ),

  private = list(
    .subscribers = list(),

    .alert = function(){
      if( length( private$.subscribers ) ){
        for( i in 1:length( private$.subscribers ) ){
          private$.subscribers[[i]]$alert( names( private$.subscribers )[[i]] )
        }
      }

      invisible( TRUE )
    },

    .alert.context = function(){
      if( length( private$.subscribers ) ){
        for( i in 1:length( private$.subscribers ) ){
          private$.subscribers[[i]]$alert.context( names( private$.subscribers )[[i]] )
        }
      }

      invisible( TRUE )
    },

    .alert.content = function(){
      if( length( private$.subscribers ) ){
        for( i in 1:length( private$.subscribers ) ){
          private$.subscribers[[i]]$alert.content( names( private$.subscribers )[[i]] )
        }
      }

      invisible( TRUE )
    }
  )
)
