library( cuRious )
library( microbenchmark )

var1 <- 1

.Call( "cuR_is_only_reference", 1+0 )


.Internal(inspect(1))
var2 <- var1
.Internal(inspect(var1))
var3 <- var1
.Internal(inspect(var3))
