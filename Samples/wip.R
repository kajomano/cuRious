library( cuRious )
library( microbenchmark )

mult <- 1

cols <- 10 * mult
subs <- c( 1 * mult + 1 , 7 * mult )

vect.x <- as.integer( rep( c( 1, 2, 3 ), times = cols / 2 ) )

tens.x.0 <- tensor$new( vect.x, 0 )
tens.p.0 <- tensor$new( vect.x, 0, copy = FALSE )
tens.w.0 <- tensor$new( NULL, 0, c( 1, 4 ), "i" )
tens.s.0 <- tensor$new( NULL, 0, c( 1, subs[[2]] - subs[[1]] + 1 ), "i" )

tens.x.3 <- tensor$new( tens.x.0, 3 )
tens.p.3 <- tensor$new( tens.p.0, 3 )
tens.w.3 <- tensor$new( tens.w.0, 3 )
tens.s.3 <- tensor$new( tens.s.0, 3 )

# Mandatory variables
stream  <- cuda.stream$new( FALSE )
context <- thrust.context$new( stream )

L0 <- thrust.table$new( tens.x.0, tens.p.0, tens.w.0, tens.s.0, subs, subs, c( 2, 4 ), context = context )
L3 <- thrust.table$new( tens.x.3, tens.p.3, tens.w.3, tens.s.3, subs, subs, c( 2, 4 ), context = context )

L0$run()
L3$run()

# tens.p.0$pull()
# tens.p.3$pull()

tens.w.0$pull()

# ITT ====
# Vmiert az utso szamot nem irja ki jol
# Allokatort lecsekkolni printekkel
# Megirni az error returnoket
tens.w.3$pull()

# tens.s.3$pull()

