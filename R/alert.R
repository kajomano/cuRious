# Alerting classes are organized this way because R6 is missing multiple
# inheritance, hax by Benedek Schultz

# .alert.recv.public ====
.alert.recv.public <- list(
  unsubscribe.flag = FALSE,

  alert = function( name ){
    self$alert.context( name )
    self$alert.content( name )
  },

  alert.context = function( name ){
    private$.add.context.changed( name )
  },

  alert.content = function( name ){
    private$.add.content.changed( name )
  }
)

# .alert.recv.private ====
.alert.recv.private <- list(
  .context.changed = NULL,
  .content.changed = NULL,

  .subscribe = function( sender, name ){
    if( !( "cuR.alert.send" %in% class( sender ) ) ){
      stop( "Invalid sender" )
    }

    sender$subscriber.add( self, name )

    private$.add.content.changed( name )
    private$.add.context.changed( name )
  },

  .unsubscribe = function( sender, name ){
    if( !( "cuR.alert.send" %in% class( sender ) ) ){
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
)

# .alert.recv.active ====
.alert.recv.active <- list(
  listener.remove = function( val ){
    if( missing( val ) ) return( private$.listener.remove )
  }
)

# .alert.send.public ====
.alert.send.public <- list(
  subscriber.add = function( subscriber, name ){
    if( !( "cuR.alert.recv" %in% class( subscriber ) ) ){
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
)

# .alert.send.private ====
.alert.send.private <- list(
  .subscribers = list(),

  .deploy = function( expr ){
    super$.deploy( expr )
    private$.alert()
  },

  .destroy = function( expr ){
    super$.destroy( expr )
    private$.alert()
  },

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

# .alert.send.active ====
.alert.send.active <- list(
  device = function( device ){
    if( missing( device ) ){
      return( private$.device )
    }else{
      super$device <- device
      private$.alert.context()
    }
  }
)

# alert.recv ====
# Pure alert recievers are not containers
.alert.recv <- R6Class(
  "cuR.alert.recv",
  public  = .alert.recv.public,
  private = .alert.recv.private,
  active  = .alert.recv.active
)

# alert.send ====
# Alert senders are also containers
.alert.send <- R6Class(
  inherit = .container,
  "cuR.alert.send",
  public  = .alert.send.public,
  private = .alert.send.private,
  active  = .alert.send.active
)

# alert.send.recv ====
.alert.send.recv <- R6Class(
  inherit = .alert.send,
  "cuR.alert.recv",
  public  = .alert.recv.public,
  private = .alert.recv.private,
  active  = .alert.recv.active
)
