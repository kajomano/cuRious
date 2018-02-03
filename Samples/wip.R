library( cuRious )
library( microbenchmark )

# TODO ====
# Templatelt transfer callok

# TODO ====
# Legyen forceolt destructor a tensorokban ha felj0nnek

# TODO ====
# Subsettelt async memcpy L3-L3

# TODO ====
# trnsfr.ptr egye meg argumentumokkal a fontos dolgokat

# TODO ====
# transfer egyen meg .ptr-eket is

test <- create.obj( c(2L,3L), level = 0, type = "double" )
test <- matrix( as.numeric(1:6), 2, 3 )
tens <- tensor$new( test )
tens$get.obj

get.type(test)
class( test )[[1]]

l <- 1000
d <- 1000
k <- 100
x <- matrix( rnorm( l*d ), l, d )
c <- x[ sample(1:l, k),]
microbenchmark( kmeans( x, c ), times = 10 )

test
destroy.obj( test )
test
