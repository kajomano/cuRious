library( cuRious )

src <- tensor$new()
dst <- tensor$new()

pip <- pipe$new( src, dst )

dst$ptr <- 2

pip$run()
