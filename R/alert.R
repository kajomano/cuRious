alert.recv <- R6Class(
  "cuR.alert.subscriber",
  public = list(
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
    .context.changed  = NULL,
    .content.changed  = NULL,
    .unsubscribe.flag = FALSE,

    .subscribe = function( sender, name ){
      if( !( "cuR.alert.sender" %in% class( sender ) ) ){
        stop( "Invalid sender" )
      }

      sender$subscriber.add( selfr, name )
    },

    .unsubscribe = function( sender ){
      if( !( "cuR.alert.sender" %in% class( sender ) ) ){
        stop( "Invalid sender" )
      }

      private$.unsubscribe.flag <- TRUE
      sender$subscriber.remove()
      private$.unsubscribe.flag <- FALSE
    },

    .add.content.changed = function( name ){
      if( ( name %in% private$.content.changed ) ){
        private$.content.changed <- c( private$.content.changed, name )
      }
    },

    .add.context.changed = function( name ){
      if( ( name %in% private$.context.changed ) ){
        private$.context.changed <- c( private$.context.changed, name )
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
      attr( subscriber, name ) <- name
      private$.subscribers <- c( private$.subscribers, list( subscriber ) )

      invisible( self )
    },

    subscriber.remove = function(){
      match <- sapply( private$.subscribers, `[[`, "unsubscribe.flag" )
      private$.subscribers <- private$.subscribers[ !match ]
      invisible( self )
    }
  ),

  private = list(
    .subscribers = list(),

    .alert = function(){
      for( subscriber in private$.subscribers ){
        subscriber$alert( attr( subscriber, name ) )
      }

      invisible( TRUE )
    },

    .alert.context = function(){
      for( subscriber in private$.subscribers ){
        subscriber$alert.context( attr( subscriber, name ) )
      }

      invisible( TRUE )
    },

    .alert.content = function(){
      for( subscriber in private$.subscribers ){
        subscriber$alert.content( attr( subscriber, name ) )
      }

      invisible( TRUE )
    }
  )
)
