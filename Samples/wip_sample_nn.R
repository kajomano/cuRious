library( cuRious )
library( microbenchmark )

basic.nn <- nn$new()
basic.nn$layers <- list( layer.input$new(1) )
basic.nn$layers[[1]]$set.nn <- basic.nn
