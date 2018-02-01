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

test <- create.obj( c(1L,1L), level = 0, type = "numeric" )
tens <- tensor$new( test )

test <- create.obj( c(3L,3L), level = 0, type = "logical" )

get.type(test)
class( test )[[1]]

test
destroy.obj( test )
test
